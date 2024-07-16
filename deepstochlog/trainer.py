from time import time
from typing import Callable, Tuple, List, Union
from pathlib import Path
import json
from transformers import BertTokenizer, BertForSequenceClassification,T5Tokenizer, T5ForConditionalGeneration

import pandas as pd
import torch
from pandas import DataFrame

from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.network import Network, NetworkStore
from deepstochlog.term import Term, List
from deepstochlog.utils import create_model_accuracy_calculator


class DeepStochLogLogger:
    def log_header(self, accuracy_tester_header):
        pass

    def log(
        self,
        epoch: int,
        batch_idx: int,
        total_loss: float,
        instances_since_last_log: int,
        accuracy_tester: str,
    ):
        raise NotImplementedError()

    def print(
        self,
        line: str,
    ):
        print(line)


class PrintLogger(DeepStochLogLogger):
    def log_header(self, accuracy_tester_header):
        print("Epoch\tBatch\tLoss\t\t" + accuracy_tester_header)

    def log(
        self,
        epoch: int,
        batch_idx: int,
        total_loss: float,
        instances_since_last_log: int,
        accuracy_tester: str,
    ):
        print(
            "{:>5}\t{:>5}\t{:.5f}\t\t{}".format(
                epoch,
                batch_idx,
                float(total_loss) / instances_since_last_log,
                accuracy_tester,
            )
        )


class PrintFileLogger(DeepStochLogLogger):
    def __init__(self, filepath):
        self.filepath = filepath

    def _write_and_print(self, line):
        with open(self.filepath, "a") as f:
            f.write(line)
            f.write("\n")
        print(line)

    def log_header(self, accuracy_tester_header):
        line = "Epoch\tBatch\tLoss\t\t" + accuracy_tester_header
        self._write_and_print(line)

    def print(self, line):
        self._write_and_print(line)

    def log(
        self,
        epoch: int,
        batch_idx: int,
        total_loss: float,
        instances_since_last_log: int,
        accuracy_tester: str,
    ):

        line = "{:>5}\t{:>5}\t{:.5f}\t\t{}".format(
            epoch,
            batch_idx,
            float(total_loss) / instances_since_last_log,
            accuracy_tester,
        )
        self._write_and_print(line)


class PandasLogger(DeepStochLogLogger):
    def __init__(self):
        self.df: DataFrame = DataFrame()

    def log_header(self, accuracy_tester_header):
        columns = ["Epoch", "Batch", "Loss"] + accuracy_tester_header.split("\t")
        self.df = DataFrame(data=[], columns=columns)

    def log(
        self,
        epoch: int,
        batch_idx: int,
        total_loss: float,
        instances_since_last_log: int,
        accuracy_tester: str,
    ):
        to_append: List[Union[str, float, int]] = [
            epoch,
            batch_idx,
            total_loss / instances_since_last_log,
        ]
        to_append.extend(
            [float(el) for el in accuracy_tester.split("\t") if len(el.strip()) > 0]
        )
        series = pd.Series(to_append, index=self.df.columns)
        self.df = self.df.append(series, ignore_index=True)

    def get_last_result(self):
        return self.df.iloc[[-1]]


print_logger = PrintLogger()

def update_nnstore(sources, networks, is_train, task):
    if is_train:
        dbs = list(set(query.arguments[1].functor.replace("'", "") for query in sources))
    else:
        dbs = [db.functor.replace("'", "") for db in sources]
    tables = []
    with open("src/" + task + "/" + "schema_" + task + ".json", 'r') as f:
        schema = json.load(f)
    tab_domains = []
    col_domains = []
    for i in range(len(dbs)):
        db = dbs[i]
        schema_parts = schema[db].split("\n")
        table_domain = schema_parts[0][schema_parts[0].index("[")+1:schema_parts[0].index("]")].split(", ")
        table_domain = [token.replace("'", "") for token in list(table_domain)]
        column_domain = []
        tables.append(table_domain)
        for part in schema_parts[2:]:
            if part:
                column_domain.append(part[part.index("[") + 1:part.index("]")].split(", "))
        tab_domains.append([Term(token) for token in table_domain])
        for elem in column_domain:
            for i in range(len(elem)):
                elem[i] = Term(elem[i].replace("'", ""))
        col_domains.append(column_domain)
    networks.get_network("tab_picker").update_index(tab_domains, dbs)
    networks.get_network("col_picker").update_index(col_domains, dbs, tables)



class DeepStochLogTrainer:
    def __init__(
        self,
        logger: DeepStochLogLogger = print_logger,
        log_freq: int = 50,
        # accuracy_tester: Tuple[str, Callable[[], str]] = (),
        accuracy_tester=None,
        test_query=None,
        print_time=False,
        allow_zero_probability_examples=False,
    ):
        self.logger = logger
        self.log_freq = log_freq
        if accuracy_tester is None:
            self.accuracy_tester_header = None
            self.accuracy_tester_fn = None
        else:
            self.accuracy_tester_header, self.accuracy_tester_fn = accuracy_tester
        self.test_query = test_query
        self.print_time = print_time
        self.allow_zero_probability_examples = allow_zero_probability_examples


    def train(
        self,
        # model: DeepStochLogModel,
        model,
        optimizer,
        dataloader: DataLoader,
        epochs: int,
        networks,
        device,
        verbose,
        verbose_building,
        test_dataloader = None,
        start_time = None,
        process_query = False,
        mask_index = 0,
        mask_term = None,
        test_dbs = None,
        refresh_model = False,
        proportion = None,
        epsilon=1e-8,
        task = "task1",
    ):
        # Test the performance using the test query
        # if self.test_query is not None:
        #     self.test_query()

        batch_idx = 0
        total_loss = 0
        instances_since_last_log = 0
        if self.accuracy_tester_header is not None:
            self.logger.log_header(self.accuracy_tester_header)
        test_model = None
        for epoch in range(epochs):
            # print(epoch)
            for batch in dataloader:
                # print(batch)
                training_start = time()
                if model is None:
                    queries = set()
                    for elem in batch:
                        query = elem.calculate_query(masked_generation_output=False)
                        if query not in queries:
                            queries.add(query)

                    update_nnstore(queries, networks, True, task)

                    grounding_start_time = time()

                    model = DeepStochLogModel.from_file(
                        # file_location=str((root_path / "wap.pl").absolute()),
                        file_location="src/" + task + "/" + task + ".pl",
                        schema_location="src/" + task + "/" + "schema_" + task + ".json",
                        query=queries,
                        networks=networks,
                        device=device,
                        verbose=verbose_building,
                        test_dbs = test_dbs,
                        proportion = proportion,
                        # normalization="FULL_NORM",
                    )
                    grounding_time = time() - grounding_start_time
                    # if verbose:
                    #     print("Grounding the program took {:.3} seconds".format(grounding_time))

                # Cross-Entropy (CE) loss
                probabilities = model.predict_sum_product(batch, process_query)
                if self.allow_zero_probability_examples:
                    targets = torch.as_tensor(
                        [el.probability for el in batch], device=model.device
                    )
                    losses = -(
                        targets * torch.log(probabilities + epsilon)
                        + (1.0 - targets) * torch.log(1.0 - probabilities + epsilon)
                    )
                else:
                    losses = -torch.log(probabilities + epsilon)

                loss = torch.mean(losses)
                loss.backward()

                # Step optimizer for learning
                optimizer.step()
                optimizer.zero_grad()

                # Save loss for printing
                total_loss += float(loss)
                instances_since_last_log += len(batch)

                if refresh_model:
                    model = None
                batch_idx += 1
                # print(
                # "\nTraining a batch of size {} took {:.2f} seconds".format(
                # dataloader.batch_size, time() - training_start
                # ))

            update_nnstore(test_dbs, networks, False, task)
            if test_model is None:
                queries = set()
                for test_batch in test_dataloader:
                    for elem in test_batch:
                        query = elem.calculate_query(masked_generation_output=True)
                        if query not in queries:
                            queries.add(query)
                    break
                test_model = DeepStochLogModel.from_file(
                    # file_location=str((root_path / "wap.pl").absolute()),
                    file_location="src/"+task+"/"+task+".pl",
                    schema_location="src/"+task+"/"+"schema_"+task+".json",
                    query=queries,
                    networks=networks,
                    device=device,
                    verbose=verbose_building,
                    test_dbs = test_dbs,
                    proportion = proportion,
                    # normalization="FULL_NORM",
                )

            start_time = time()
            calculate_model_accuracy = create_model_accuracy_calculator(
                model=test_model,
                test_dataloader=test_dataloader,
                start_time=start_time,
                val_dataloader=None,
                most_probable_parse_accuracy=False,
            )
            accuracy_tester_header, accuracy_tester_fn = calculate_model_accuracy
            # Print the loss
            self.logger.log_header(accuracy_tester_header)
            self.logger.log(
                epoch,
                batch_idx,
                total_loss,
                instances_since_last_log,
                accuracy_tester_fn(mask_index=mask_index, mask_term=mask_term, process_query=process_query, test_dbs = test_dbs),
            )
            total_loss = 0
            instances_since_last_log = 0

        end_time = time() - training_start
        if self.print_time:
            self.logger.print(
                "\nTraining {} epoch (totalling {} batches of size {}) took {:.2f} seconds".format(
                    epochs, epochs * len(dataloader), dataloader.batch_size, end_time
                )
            )

        # Test the performance on the first test query again
        # if self.test_query is not None:
        #     self.test_query()

        return end_time

    def should_log(self, batch_idx, dataloader, epoch, epochs):
        return batch_idx % self.log_freq == 0 or (
            epoch == epochs - 1 and batch_idx % len(dataloader) == len(dataloader) - 1
        )
