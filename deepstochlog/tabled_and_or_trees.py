import itertools
from functools import lru_cache
from typing import Dict, Tuple, List, Callable, Iterable

import torch
from torch import Tensor

from deepstochlog.context import Context, ContextualizedTerm
from deepstochlog.networkevaluation import NetworkEvaluations, RequiredEvaluation
from deepstochlog import term
from deepstochlog.term import Term, List
from deepstochlog.inferences import NNLeafDescendantsRetriever, TermLeafDescendantsRetriever


def query_process(t):
    try:
        res = Term(
            t.functor,
            Term(t.arguments[0].functor.replace("'", "")),
            Term(t.arguments[1].functor.replace("'", "")),
            term.List(*[Term(elem.functor.replace("'", "")) for elem in t.arguments[2]]),
        )
    except:
        res = Term(
            t.functor,
            Term(t.arguments[0].functor.replace("'", "")),
            Term(t.arguments[1].functor.replace("'", "")),
            term.List(*[Term(elem.functor.replace("'", "")) for elem in [t.arguments[2]]]),
        )
    return res

class TabledAndOrTrees:
    """
    Represents the grounded and/or tree, used for calculating the probabilities easily
    """

    def __init__(
        self,
        and_or_tree: Dict[Term, "LogicNode"],
        terms_grounder: Callable[[List[Term]], Dict[Term, "LogicNode"]] = None,
    ):
        self._and_or_tree = and_or_tree
        self._terms_grounder = terms_grounder

    def has_and_or_tree(self, t: Term):
        return t in self._and_or_tree

    def get_and_or_tree(self, t: Term, process_query=False):

        if not self.has_and_or_tree(t):
            self.ground_all([t])
        if process_query and t.functor == "query" and "'" in t.arguments[0].functor:
            t_processed = query_process(t)
            return self._and_or_tree[t_processed]
        else:
            return self._and_or_tree[t]


    def ground_all(self, terms: List[Term]):
        if self._terms_grounder is None:
            raise RuntimeError(
                "Did not contain a term grounder, so could not ground missing terms",
                [str(t) for t in terms],
            )
        new_elements = self._terms_grounder(terms)
        # add new (key, value) pairs to dict
        self._and_or_tree.update(new_elements)

    def calculate_required_evaluations(
        self, contextualized_term: ContextualizedTerm, process_query=False,
    ) -> List[RequiredEvaluation]:
        nn_leafs = list()
        handled_terms = set()
        term_queue = [contextualized_term.term]
        # Get all descendent nn_leafs
        while len(term_queue) > 0:
            t = term_queue.pop()
            if t not in handled_terms:
                handled_terms.add(t)
                if process_query and t.functor == "query":
                    t_processed = query_process(t)
                    tree = self.get_and_or_tree(t_processed)
                else:
                    tree = self.get_and_or_tree(t)
                # Extend with all neural leafs
                nn_leafs.extend(
                    tree.accept_visitor(visitor=NNLeafDescendantsRetriever())
                )
                # Add all term leafs as terms to handle
                term_queue.extend(
                    [
                        t.term
                        for t in tree.accept_visitor(
                            visitor=TermLeafDescendantsRetriever()
                        )
                    ]
                )

        # Turn every leaf into a required evaluation
        return [
            nn_leaf.to_required_evaluation(contextualized_term.context)
            for nn_leaf in nn_leafs
        ]


class LogicProbabilityEvaluator:
    def __init__(
        self,
        trees: TabledAndOrTrees,
        network_evaluations: NetworkEvaluations,
        device=None,
    ):
        self.device = device
        self.trees = trees
        self.network_evaluations = network_evaluations

    @lru_cache()
    def accept_term_visitor(self, t: Term, visitor: "LogicNodeVisitor", process_query=False):
        if process_query and t.functor == "query":
            return self.trees.get_and_or_tree(query_process(t)).accept_visitor(visitor=visitor)
        else:
            return self.trees.get_and_or_tree(t).accept_visitor(visitor=visitor)

    def evaluate_neural_network_probability(
        self,
        network: str,
        input_arguments: Tuple,
        index: int,
        context: Context = None,
    ) -> Tensor:
        return self.network_evaluations.get_evaluation_result(
            context=context, network_name=network, input_args=input_arguments
        )[index]
