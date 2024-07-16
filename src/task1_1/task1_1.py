import torch
from torch.optim import Adam
from transformers import BertForSequenceClassification, T5ForConditionalGeneration
from time import time
import gc
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from deepstochlog.network import Network, NetworkStore
from deepstochlog.utils import set_fixed_seed
from deepstochlog.dataloader import DataLoader
from deepstochlog.term import Term
from deepstochlog.trainer import DeepStochLogTrainer, print_logger
from src.task1.spider_data import SpiderDataset



def run(
    epochs=2,
    train_batch_size=8,
    val_batch_size=8,
    lr=1e-3,
    task = "task1",
    train_size=None,
    val_size=None,
    log_freq=50,
    logger=print_logger,
    seed=None,
    verbose=True,
    verbose_building=True,
):

    #start_time = time()
    set_fixed_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    t5_type = "t5-small"
    tab_picker = Network(
        "tab_picker",
        T5ForConditionalGeneration.from_pretrained(t5_type),
        index_list=None,
        network_type= t5_type,
        concat_tensor_input=False,
    )
    col_picker = Network(
        "col_picker",
        T5ForConditionalGeneration.from_pretrained(t5_type),
        index_list=None,
        network_type= t5_type,
        concat_tensor_input=False,
    )
    select_switcher = Network(
        "select_switcher",
        T5ForConditionalGeneration.from_pretrained(t5_type),
        index_list=[Term(str(e)) for e in range(10)],
        network_type=t5_type,
        concat_tensor_input=False,
    )
    groupby_switcher = Network(
        "groupby_switcher",
        T5ForConditionalGeneration.from_pretrained(t5_type),
        index_list=[Term(str(e)) for e in range(2)],
        network_type=t5_type,
        concat_tensor_input=False,
    )
    orderby_switcher = Network(
        "orderby_switcher",
        T5ForConditionalGeneration.from_pretrained(t5_type),
        index_list=[Term(str(e)) for e in range(3)],
        network_type=t5_type,
        concat_tensor_input=False,
    )
    asc_switcher = Network(
        "asc_switcher",
        T5ForConditionalGeneration.from_pretrained(t5_type),
        index_list=[Term(str(e)) for e in range(2)],
        network_type=t5_type,
        concat_tensor_input=False,
    )
    networks = NetworkStore(select_switcher, tab_picker, col_picker, groupby_switcher, orderby_switcher, asc_switcher)


    train_data = SpiderDataset(
        split="train",
        task=task,
        size=train_size,
    )
    val_data = SpiderDataset(
        split="test",
        task=task,
        size=val_size,
    )
    val_dbs = val_data.getdbs()


    train_dataloader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False,
    )

    optimizer = Adam(networks.get_all_net_parameters(), lr=lr)
    optimizer.zero_grad()

    trainer = DeepStochLogTrainer(
        log_freq=log_freq,
        accuracy_tester=None,
        logger=logger,
        test_query=None,
        print_time=verbose,
    )
    train_time = trainer.train(
        model=None,
        optimizer=optimizer,
        dataloader=train_dataloader,
        epochs=epochs,
        networks=networks,
        device=device,
        verbose=verbose,
        verbose_building=verbose_building,
        process_query=True,
        test_dataloader=val_dataloader,
        mask_index=2,
        test_dbs = val_dbs,
        refresh_model = True,
        task = task,
    )


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    run(
        epochs=15,
        seed=42,
        log_freq=50,
        train_batch_size=4,
        val_batch_size=1,
        task = "task1_1"
        # verbose_building=False,
    )
