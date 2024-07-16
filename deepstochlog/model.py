import typing
from typing import Iterable, List, Union, Collection
import json
from time import time

import torch

from deepstochlog.context import ContextualizedTerm
from deepstochlog.network import NetworkStore
from deepstochlog.logic import Or, And, TermLeaf, LogicNode
from deepstochlog.networkevaluation import RequiredEvaluation, NetworkEvaluations
from deepstochlog.tabled_and_or_trees import TabledAndOrTrees, LogicProbabilityEvaluator
from deepstochlog.tabled_tree_builder import TabledAndOrTreeBuilder
from deepstochlog.term import Term
from deepstochlog.parser import parse_rules
from deepstochlog.inferences import (
    SumProductVisitor,
    MaxProductVisitor,
    TermLeafDescendantsRetriever,
)


def remove_duplicates(input_list):
    unique_elements = []
    seen_elements = set()

    for element in input_list:
        if element not in seen_elements:
            unique_elements.append(element)
            seen_elements.add(element)

    return unique_elements


class DeepStochLogModel:

    NO_NORM = "NO_NORM"
    LEN_NORM = "LEN_NORM"
    FULL_NORM = "FULL_NORM"

    def __init__(
        self,
        trees: TabledAndOrTrees,
        neural_networks: NetworkStore,
        normalization=None,
        device=None,
    ):
        self.neural_networks = neural_networks
        self.device = device
        self.trees = trees
        self.normalization = self._set_normalization(normalization)
        if device is not None:
            self.to_device(device)

    def _set_normalization(self, normalization):
        if normalization is None:
            normalization = DeepStochLogModel.NO_NORM
        if normalization not in (
            DeepStochLogModel.NO_NORM,
            DeepStochLogModel.LEN_NORM,
            DeepStochLogModel.FULL_NORM,
        ):
            raise ValueError("Normalization %s unknown." % str(normalization))
        return normalization

    def to_device(self, *args, **kwargs):
        """ Allows to put the networks on the GPU """
        self.neural_networks.to_device(*args, **kwargs)
        self.device = args[0]

    def get_all_net_parameters(self):
        return self.neural_networks.get_all_net_parameters()

    def compute_normalization_constant(
        self, probability_evaluator, contextualized_term
    ):

        if self.normalization == DeepStochLogModel.NO_NORM:
            Z = 1.0
        elif self.normalization == DeepStochLogModel.LEN_NORM:
            raise NotImplementedError("Length based normalization not implemented")
        else:
            Z = probability_evaluator.accept_term_visitor(
                t=contextualized_term.term.mask_generation_output(),
                visitor=SumProductVisitor(
                    probability_evaluator=probability_evaluator,
                    context=contextualized_term.context,
                ),
            )
        return Z

    def predict_sum_product(self, batch: Iterable[ContextualizedTerm], process_query=False, model_eval=False, test_dbs = None) -> torch.Tensor:
        probability_evaluator = self.create_probability_evaluator(batch, process_query, model_eval=model_eval, test_dbs = test_dbs)
        tensors = []
        for contextualized_term in batch:
            p = probability_evaluator.accept_term_visitor(
                t=contextualized_term.term,
                visitor=SumProductVisitor(
                    probability_evaluator=probability_evaluator,
                    context=contextualized_term.context,
                ),
                process_query = process_query,
            )
            # p = probability_evaluator.evaluate_term_sum_product_probability(
            #     term=contextualized_term.term, context=contextualized_term.context
            # )
            Z = self.compute_normalization_constant(
                probability_evaluator, contextualized_term
            )
            tensors.append(p / Z)

        return torch.stack(tensors)

    def predict_max_product_parse(
        self, batch: Iterable[ContextualizedTerm]
    ) -> typing.List[typing.Tuple[torch.Tensor, Iterable[LogicNode]]]:
        probability_evaluator = self.create_probability_evaluator(batch)

        predictions: typing.List[typing.Tuple[torch.Tensor, Iterable[LogicNode]]] = [
            TermLeaf(term=contextualized_term.term).accept_visitor(
                visitor=MaxProductVisitor(
                    probability_evaluator=probability_evaluator,
                    context=contextualized_term.context,
                ),
            )
            # probability_evaluator.evaluate_term_leaf_max_product_probability(
            #     term_leaf=TermLeaf(contextualized_term.term),
            #     context=contextualized_term.context,
            # )
            for contextualized_term in batch
        ]
        return predictions

    def create_probability_evaluator(self, batch: Iterable[ContextualizedTerm], process_query=False, model_eval=False, test_dbs = None):
        # not use this to avoid
        required_evaluations: List[RequiredEvaluation] = remove_duplicates(
            [
                re
                for ct in batch
                for re in self.trees.calculate_required_evaluations(
                    contextualized_term=ct, process_query=process_query
                )
            ]
        )


        # required_evaluations: List[RequiredEvaluation] = [
        #         re
        #         for ct in batch
        #         for re in self.trees.calculate_required_evaluations(
        #             contextualized_term=ct, process_query=process_query
        #         )
        # ]


        # Evaluate all required evaluations on the neural networks
        network_evaluations: NetworkEvaluations = (
            NetworkEvaluations.from_required_evaluations(
                required_evaluations=required_evaluations,
                networks=self.neural_networks,
                device=self.device,
                batch = batch,
                model_eval = model_eval,
                test_dbs=test_dbs
            )
        )
        # Calculate the probabilities using the evaluated networks results.
        probability_evaluator = LogicProbabilityEvaluator(
            trees=self.trees,
            network_evaluations=network_evaluations,
            device=self.device,
        )
        return probability_evaluator

    def calculate_probability(self, contextualized_term: ContextualizedTerm):
        """ For easy access from test cases, and demonstrating. Not really used in core model """
        probability_evaluator = self.create_probability_evaluator([contextualized_term])
        p = probability_evaluator.accept_term_visitor(
            term=contextualized_term.term,
            visitor=SumProductVisitor(
                probability_evaluator=probability_evaluator,
                context=contextualized_term.context,
            ),
        )
        Z = self.compute_normalization_constant(
            probability_evaluator, contextualized_term
        )
        return p / Z

    @staticmethod
    def from_string(
        program_str: str,
        networks: NetworkStore,
        query: Union[Term, Iterable[Term]] = None,
        device=None,
        verbose=False,
        add_default_zero_probability=False,
        prolog_facts: str = "",
        tabling=True,
        normalization=None,
    ):
        parse_start = time()
        deepstochlog_rules = parse_rules(program_str)
        deepstochlog_rules, extra_networks = deepstochlog_rules.remove_syntactic_sugar()
        parse_time = time() - parse_start
        # print("Parsing the program took {:.3} seconds".format(parse_time))

        networks = networks + extra_networks

        # covert ndgc to prolog, add tabling header
        builder = TabledAndOrTreeBuilder(
            deepstochlog_rules,
            verbose=verbose,
            prolog_facts=prolog_facts,
            tabling=tabling,
        )

        tabled_and_or_trees = builder.build_and_or_trees(
            networks=networks,
            queries=query,
            add_default_zero_probability=add_default_zero_probability,
        )


        return DeepStochLogModel(
            trees=tabled_and_or_trees,
            neural_networks=networks,
            device=device,
            normalization=normalization,
        )

    @staticmethod
    def from_file(
        file_location: str,
        networks: NetworkStore,
        schema_location: str = None,
        query: Union[Term, Iterable[Term]] = None,
        test_dbs = None,
        device=None,
        add_default_zero_probability=False,
        verbose=False,
        prolog_facts: str = "",
        tabling=True,
        proportion=None,
        normalization=None,
    ) -> "DeepStochLogModel":
        with open(file_location) as grammar_f:
            grammar_l = grammar_f.readlines()
        db_tab = []
        tab = []
        if schema_location:
            with open(schema_location) as schema_f:
                all_schema = json.load(schema_f)
                for q in query:
                    db_id = q.arguments[1].functor[1:-1]
                    for s in all_schema[db_id].split("\n"):
                        if s:
                            if s.startswith("database_tables"):
                                db_tab.append(s)
                            elif s.startswith("table"):
                                tab.append(s)
                            else:
                                continue
                if test_dbs:
                    for db in test_dbs:
                        db_id = db.functor[1:-1]
                        for s in all_schema[db_id].split("\n"):
                            if s:
                                if s.startswith("database_tables"):
                                    db_tab.append(s)
                                elif s.startswith("table"):
                                    tab.append(s)
                                else:
                                    continue
        # could encode data bias in grammar file
        if proportion is None:
            lines = db_tab + tab + grammar_l
        else:
            bias_l = []
            for i in range(len(proportion)):
                bias_l.append(str(proportion[i])+"::"+"selection_bias("+str(i)+") --> [].")
            lines = db_tab + tab + bias_l + grammar_l

        return DeepStochLogModel.from_string(
            "\n".join(lines),
            query=query,
            networks=networks,
            device=device,
            verbose=verbose,
            add_default_zero_probability=add_default_zero_probability,
            prolog_facts=prolog_facts,
            tabling=tabling,
            normalization=normalization,
        )

    def mask_and_get_direct_proof_possibilities(self, term: Term, mask_index=0, mask_term=None, process_query=False) -> List[Term]:
        if term.can_mask_generation_output():
            return self.get_direct_proof_possibilities(term.mask_generation_output(mask_index=mask_index, mask_term=mask_term), process_query=process_query)
        else:
            return [term]

    def get_direct_proof_possibilities(self, term: Term, process_query=False) -> List[Term]:
        term_tree: LogicNode = self.trees.get_and_or_tree(term, process_query=process_query)
        term_leafs: Iterable[TermLeaf] = term_tree.accept_visitor(
            visitor=TermLeafDescendantsRetriever()
        )
        terms = [tl.term for tl in term_leafs]
        return terms

    def get_direct_contextualized_proof_possibilities(
        self, ct: ContextualizedTerm, mask_index=0, mask_term=None, process_query = False,
    ) -> List[ContextualizedTerm]:
        return [
            ContextualizedTerm(context=ct.context, term=t)
            for t in self.mask_and_get_direct_proof_possibilities(ct.term, mask_index=mask_index, mask_term=mask_term, process_query=process_query)
        ]
