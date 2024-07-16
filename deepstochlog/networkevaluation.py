from collections import defaultdict
from typing import Dict, Tuple, Iterable, List
from transformers import BertTokenizer, T5Tokenizer
import json

import torch
from torch import Tensor
import torch.nn.functional as F

from deepstochlog.term import Term
from deepstochlog.context import Context
from deepstochlog.network import NetworkStore


class RequiredEvaluation:
    def __init__(self, context: Context, network_name: str, input_args: Tuple):
        self.context = context
        self.network_name = network_name
        self.input_args = input_args

    def prepare(self) -> "PreparedEvaluation":
        """ Prepares the evaluation by mapping the input variables to the right tensors """
        mapped_input_args = self.context.get_all_tensor_representations(self.input_args)
        return PreparedEvaluation(self, mapped_input_args)

    def __str__(self):
        return (
            "RequiredEvaluation(<context>, "
            + self.network_name
            + ", "
            + str(self.input_args)
            + ")"
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, RequiredEvaluation):
            return (
                self.context == other.context
                and self.network_name == other.network_name
                and self.input_args == other.input_args
            )
        return False

    def __hash__(self):
        return hash((self.context, self.network_name, self.input_args))


class PreparedEvaluation:
    def __init__(
        self, required_evaluation: RequiredEvaluation, mapped_input_args: List[Tensor]
    ):
        self.required_evaluation = required_evaluation
        self.mapped_input_args = mapped_input_args

    def has_tensors(self):
        return len(self.mapped_input_args) > 0

    def __str__(self):
        return (
            "PreparedEvaluation("
            + str(self.required_evaluation)
            + ", tensor_list(len:"
            + str(len(self.mapped_input_args))
            + "))"
        )

    def __repr__(self):
        return str(self)


def extract_input_arguments(prepared_evaluations: Iterable[PreparedEvaluation]):
    return [
        torch.cat(pe.mapped_input_args, 0) if pe.has_tensors() else torch.tensor([])
        for pe in prepared_evaluations
    ]

def build_prompt(inputs, domain, name):
    prompts = []
    for i in range(len(inputs)):
        prompt = inputs[i][0] + " "
        if "switcher" in name:
            if name == "select_switcher":
                if len(domain) == 8:
                    grammar_option = ["*", "COUNT(*)", "column", "COUNT(column)", "SUM(column)", "AVG(column)", "MIN(column)", "MAX(column)"]
                else:
                    grammar_option = ["*", "COUNT(*)", "column", "DISTINCT column", "COUNT(column)", "COUNT(DISTINCT column)", "SUM(column)", "AVG(column)",
                                      "MIN(column)", "MAX(column)"]
            elif name == "groupby_switcher":
                grammar_option = ["empty", "GROUP BY"]
            elif name == "orderby_switcher":
                grammar_option = ["empty", "ORDER BY COUNT(*)", "ORDER BY column"]
            elif name == "asc_switcher":
                grammar_option = ["ASC", "DESC"]
            for index in range(len(grammar_option)):
                prompt += "Answer " + str(index + 1) + " for " + grammar_option[index] + ", "
        else:
            if name == "col_picker":
                ref = Term(inputs[i][1]+inputs[i][2])
                if len(inputs[i]) ==4:
                    if inputs[i][3] == 0:
                        prompt += "SELECT [column], "
                    elif inputs[i][3] == 1:
                        prompt += "GROUP BY [column], "
                    elif inputs[i][3] == 2:
                        prompt += "ORDER BY [column], "
            elif name == "tab_picker":
                ref = Term(inputs[i][1])
            if ref not in domain:
                raise Exception(
                    "Index was not found, did you include the right Term list as keys? Error item: "
                    + str(ref)
                    + " "
                    + str(type(ref))
                    + ".\nPossible values: "
                    + ", ".join([str(k) for k in domain.keys()])
                )
            for key, value in domain[ref].items():
                prompt += "Answer " + str(int(value)+1) + " for " + key.functor + ", "
        prompt += "the answer should be Answer "
        prompts.append(prompt)
    return prompts

def get_target_prob(name, mappings, tokenizer, logits, device, model_eval):
    res = []
    for i in range(len(mappings)):
        if not model_eval:
            logit = logits[i][0].clone()
        else:
            logit = logits[i].clone()
        index = []
        for value in mappings[i].values():
            target_token_index = tokenizer.encode(str(int(value)+1))
            index.append(target_token_index[0])
        index = torch.tensor(index).to(device, dtype=torch.long)
        res.append(F.softmax(torch.index_select(logit, 0, index)))

    max_size = max(tensor.size(0) for tensor in res)
    padded_res = [F.pad(tensor, (0, max_size-tensor.size(0)), "constant", 0) for tensor in res]
    res = torch.stack(padded_res)

    return res


def extract_gt(name, batch, prepared_evaluations, index_mapping):
    gts = []
    mappings = []
    for i in range(len(prepared_evaluations)):
        pe = prepared_evaluations[i]
        nl = pe.mapped_input_args[0]
        db = pe.mapped_input_args[1]
        gt_list = []
        for j in range(len(batch)):
            if batch[j].term.arguments[0].functor.replace("'", "") == nl and batch[j].term.arguments[1].functor.replace("'", "") == db:
                gt_list = batch[j].term.arguments[2]
                break
        gt_str_list = [elem.functor.replace("'", "") for elem in gt_list]
        if name == "select_switcher":
            selection = gt_str_list[1]
            if len(index_mapping) == 8:
                grammar_option = ["*", "COUNT(*)", "column", "COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]
                if selection in grammar_option:
                    gt = grammar_option.index(selection)
                else:
                    gt = 2
            else:
                grammar_option = ["*", "COUNT(*)", "column", "DISTINCT", "COUNT(", "COUNT( DISTINCT", "SUM(", "AVG(", "MIN(", "MAX("]
                if selection == "COUNT(":
                    if gt_str_list[2] == "DISTINCT":
                        gt = 5
                    else:
                        gt = 4
                else:
                    if selection in grammar_option:
                        gt = grammar_option.index(selection)
                    else:
                        gt = 2
            mapping = index_mapping
        elif name == "groupby_switcher":
            if "GROUP BY" in gt_str_list:
                gt = 1
            else:
                gt = 0
            mapping = index_mapping
        elif name == "orderby_switcher":
            if "ORDER BY" in gt_str_list:
                if "COUNT(*)" in gt_str_list:
                    gt = 1
                else:
                    gt = 2
            else:
                gt = 0
            mapping = index_mapping
        elif name == "asc_switcher":
            if "DESC" in gt_str_list:
                gt = 1
            else:
                gt = 0
            mapping = index_mapping
        else:
            if name == "tab_picker":
                tab = gt_str_list[gt_str_list.index("FROM") + 1]
                ref = Term(db)
                key = Term(tab)
            else:
                tab = pe.mapped_input_args[2]
                if len(pe.mapped_input_args) > 3:
                    col_type = pe.mapped_input_args[3]
                else:
                    col_type = 0
                if int(col_type) == 0:
                    selection = gt_str_list[1:gt_str_list.index("FROM")]
                    if len(selection) == 1:
                        col = selection[0]
                    elif len(selection) == 4:
                        col = selection[2]
                    else:
                        col = selection[1]
                elif int(col_type) == 1:
                    col = gt_str_list[gt_str_list.index("GROUP BY") + 1]
                else:
                    col = gt_str_list[gt_str_list.index("ORDER BY") + 1]
                ref = Term(db+tab)
                key = Term(col)
            if ref not in index_mapping:
                raise Exception(
                    "Index was not found, did you include the right Term list as keys? Error item: "
                    + str(ref)
                    + " "
                    + str(type(ref))
                    + ".\nPossible values: "
                    + ", ".join([str(k) for k in index_mapping.keys()])
                )
            if key not in index_mapping[ref]:
                raise Exception(
                    "Index was not found, did you include the right Term list as keys? Error item: "
                    + str(key)
                    + " "
                    + str(type(key))
                    + ".\nPossible values: "
                    + ", ".join([str(k) for k in index_mapping[ref].keys()])
                )
            gt = index_mapping[ref][key]
            mapping = index_mapping[ref]
        gts.append(str(int(gt)+1))
        mappings.append(mapping)
    return gts, mappings


def test_mapping(name, require_eval, index_mapping):
    mappings = []
    for elem in require_eval:
        if "switcher" in name:
            mapping = index_mapping
        else:
            args = list(elem.required_evaluation.input_args)
            db = args[1]
            if name == "tab_picker":
                ref = db
            else:
                tab = args[2]
                ref = Term(db.functor+tab.functor)
            if ref not in index_mapping:
                raise Exception(
                    "Index was not found, did you include the right Term list as keys? Error item: "
                    + str(ref)
                    + " "
                    + str(type(ref))
                    + ".\nPossible values: "
                    + ", ".join([str(k) for k in index_mapping.keys()])
                )
            mapping = index_mapping[ref]
        mappings.append(mapping)
    return mappings


class NetworkEvaluations:
    def __init__(self):
        self.evaluations: Dict[Context, Dict[str, Dict[Tuple, Tensor]]] = defaultdict(
            lambda: defaultdict(defaultdict)
        )

    def add_evaluation_result(
        self, context: Context, network_name: str, input_args: Tuple, output: Tensor
    ):
        self.evaluations[context][network_name][input_args] = output

    def get_evaluation_result(
        self, context: Context, network_name: str, input_args: Tuple
    ) -> Tensor:
        return self.evaluations[context][network_name][input_args]

    @staticmethod
    def from_required_evaluations(
        required_evaluations: Iterable[RequiredEvaluation],
        networks: NetworkStore,
        device=None,
        batch=None,
        model_eval=False,
        test_dbs=None,
    ):
        """ Evaluates all networks for a list of required evaluations """
        per_network: Dict[str, List[PreparedEvaluation]] = defaultdict(list)
        for req in required_evaluations:
            per_network[req.network_name].append(req.prepare())

        # Evaluate on network
        network_evaluations: NetworkEvaluations = NetworkEvaluations()
        for network_name, prepared_evaluations in per_network.items():
            network = networks.get_network(network_name)

            if network.concat_tensor_input:
                all_to_evaluate: List[Tensor] = extract_input_arguments(
                    prepared_evaluations
                )

                neural_input = torch.nn.utils.rnn.pad_sequence(
                    all_to_evaluate, batch_first=True
                )

                if device:
                    neural_input = neural_input.to(device)
            else:
                neural_input = [pe.mapped_input_args for pe in prepared_evaluations]

            if "t5" in network.neural_model.__class__.__name__.lower():
                prompts = build_prompt(neural_input, network.index_mapping, network_name)
                tokenizer = T5Tokenizer.from_pretrained(network.network_type)
                if network_name == "tab_picker":
                    max_padding_len = 125
                elif network_name == "col_picker":
                    max_padding_len = 250
                else:
                    max_padding_len = 100
                inputs = tokenizer.batch_encode_plus(
                    prompts,
                    max_length=max_padding_len,
                    pad_to_max_length=True,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"].to(device, dtype=torch.long)
                input_mask = inputs["attention_mask"].to(device, dtype=torch.long)
                if not model_eval:
                    gts, mappings = extract_gt(network_name, batch, prepared_evaluations, network.index_mapping)
                    targets = tokenizer.batch_encode_plus(
                        gts,
                        return_tensors="pt",
                    )
                    labels = targets["input_ids"].to(device, dtype=torch.long)
                    label_ids =labels[:, :-1].contiguous()
                    if not network.neural_model.training:
                        network.neural_model.train()
                    outputs = network.neural_model(
                        input_ids=input_ids,
                        attention_mask=input_mask,
                        # labels = labels,
                        labels = label_ids,
                        return_dict=True,
                    )
                    logits = outputs.logits
                else:
                    network.neural_model.eval()
                    outputs = network.neural_model.generate(
                        input_ids=input_ids,
                        attention_mask=input_mask,
                        max_length=2,
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                    logits = outputs.scores[0]
                    mappings = test_mapping(network_name, prepared_evaluations, network.index_mapping)
                neural_outputs = get_target_prob(network_name, mappings, tokenizer, logits, device, model_eval)

            elif "bert" in network.neural_model.__class__.__name__.lower():
                tokenizer = BertTokenizer.from_pretrained(network.network_type)
                max_padding_len = 30
                inputs = tokenizer.batch_encode_plus(
                    [elem[0] for elem in neural_input],
                    max_length=max_padding_len,
                    pad_to_max_length=True,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"].to(device, dtype=torch.long)
                input_mask = inputs["attention_mask"].to(device, dtype=torch.long)
                if not model_eval:
                    if not network.neural_model.training:
                        network.neural_model.train()
                else:
                    network.neural_model.eval()
                outputs = network.neural_model(
                        input_ids=input_ids,
                        attention_mask=input_mask,
                        return_dict=True,
                )
                neural_outputs = F.softmax(outputs.logits, dim=1)
            else:
                neural_outputs = network.neural_model(neural_input)

            # Store result
            required_evaluations = [
                pe.required_evaluation for pe in prepared_evaluations
            ]
            for re, output in zip(required_evaluations, neural_outputs):
                network_evaluations.add_evaluation_result(
                    context=re.context,
                    network_name=re.network_name,
                    input_args=re.input_args,
                    output=output,
                )
        return network_evaluations
