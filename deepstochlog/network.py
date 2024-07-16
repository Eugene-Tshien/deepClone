from typing import List

import torch.nn as nn

from deepstochlog.term import Term


class Network(object):
    def __init__(
        self,
        name: str,
        neural_model: nn.Module,
        # index_list: List[Term],
        index_list,
        network_type = None,
        concat_tensor_input=True,
    ):
        self.name = name
        self.neural_model = neural_model
        self.computation_graphs = dict()
        self.network_type = network_type
        self.index_list = index_list
        self.index_mapping = dict()
        if index_list is not None:
            for i, elem in enumerate(index_list):
                self.index_mapping[elem] = i
        self.concat_tensor_input = concat_tensor_input

    def term2idx(self, term: Term, ref = None) -> int:
        #TODO(giuseppe) index only with the functor

        if not ref:
            col_dict = self.index_mapping
        else:
            if ref not in self.index_mapping:
                raise Exception(
                    "Index was not found, did you include the right Term list as keys? Error item: "
                    + str(ref)
                    + " "
                    + str(type(ref))
                    + ".\nPossible values: "
                    + ", ".join([str(k) for k in self.index_mapping.keys()])
                )
            col_dict = self.index_mapping[ref]
        key = Term(str(term.functor))
        if key not in col_dict:
            raise Exception(
                "Index was not found, did you include the right Term list as keys? Error item: "
                + str(term)
                + " "
                + str(type(term))
                + ".\nPossible values: "
                + ", ".join([str(k) for k in col_dict.keys()])
            )
        return col_dict[key]

    def idx2term(self, index: int) -> Term:
        return self.index_list[index]


    def update_index(self, new_list, dbs=None, tables=None):
        self.index_list = new_list
        if type(new_list[0]) == list:
            self.index_mapping = dict()
            if not tables:
                for index in range(len(new_list)):
                    list_d0 = new_list[index]
                    db = dbs[index].replace("'", "")
                    key = Term(db)
                    self.index_mapping[key] = dict()
                    for i, elem in enumerate(list_d0):
                        self.index_mapping[key][elem] = i
            else:
                # if type(new_list[0][0]) == list:
                for index in range(len(new_list)):
                    for index1 in range(len(new_list[index])):
                        list_d1 = new_list[index][index1]
                        db = dbs[index].replace("'", "")
                        tab = tables[index][index1].replace("'", "")
                        key = Term(db+tab)
                        self.index_mapping[key] = dict()
                        for i, elem in enumerate(list_d1):
                            self.index_mapping[key][elem] = i
                # else:
                #     for index in range(len(new_list)):
                #         list_d0 = new_list[index]
                #         db = dbs[index].replace("'", "")
                #         tab = tables[index].replace("'", "")
                #         key = Term(db + tab)
                #         self.index_mapping[key] = dict()
                #         for i, elem in enumerate(list_d0):
                #             self.index_mapping[key][elem] = i

        else:
            self.index_mapping = dict()
            for i, elem in enumerate(new_list):
                self.index_mapping[elem] = i


    def to(self, *args, **kwargs):
        self.neural_model.to(*args, **kwargs)


class NetworkStore:
    def __init__(self, *networks: Network):
        self.networks = dict()
        for n in networks:
            self.networks[n.name] = n

    def get_network(self, name: str) -> Network:
        return self.networks[name]

    def to_device(self, *args, **kwargs):
        for network in self.networks.values():
            network.to(*args, **kwargs)

    def get_all_net_parameters(self):
        all_parameters = list()
        for network in self.networks.values():
            all_parameters.extend(network.neural_model.parameters())
        return all_parameters

    def __add__(self, other: "NetworkStore"):
        return NetworkStore(
            *(list(self.networks.values()) + list(other.networks.values()))
        )
