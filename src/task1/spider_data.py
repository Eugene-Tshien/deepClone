import os
import json
from typing import Union
from deepstochlog.dataset import ContextualizedTermDataset
from deepstochlog.context import ContextualizedTerm, Context
from deepstochlog.term import Term


class SpiderDataset(ContextualizedTermDataset):
    def __init__(
        self,
        split: str = "train",
        task: str = "task1",
        size: int = None,
    ):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        data_file_path = os.path.join(data_dir, split+"_"+task+".json")
        with open(data_file_path, "r") as file:
            all_samples = json.load(file)

        if size is None:
            size = len(all_samples)

        self.ct_term_dataset = []
        self.dbs = set()
        for idx in range(0, size):
            sample = all_samples[idx]
            self.dbs.add(Term(sample["db"]))
            sample_term = create_term(sample)
            self.ct_term_dataset.append(sample_term)

    def __len__(self):
        return len(self.ct_term_dataset)

    def __getitem__(self, item: Union[int, slice]):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        return self.ct_term_dataset[item]

    def getdbs(self):
        return list(self.dbs)



def create_term(sample) -> ContextualizedTerm:

    return ContextualizedTerm(
        context=Context(
            {Term("NL"): sample["question"]},
            map_default_to_term=True,
        ),
        term=Term(
            "query",
            Term(sample["question"]),
            Term(sample["db"]),
            [Term(token) for token in sample["query_gt"]],
        ),
    )