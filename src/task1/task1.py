import argparse
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
from spider_data import SpiderDataset



def run(
    epochs=2,
    train_batch_size=8,
    val_batch_size=8,
    lr=1e-3,
    ss_type = "t5",
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
    if ss_type == "bert":
        bert_model = "bert-base-uncased"
        select_switcher = Network(
            "select_switcher",
            BertForSequenceClassification.from_pretrained(bert_model, num_labels=8),
            index_list=[Term(str(e)) for e in range(8)],
            network_type = bert_model,
            concat_tensor_input=False,
        )
    else:
        select_switcher = Network(
            "select_switcher",
            T5ForConditionalGeneration.from_pretrained(t5_type),
            index_list=[Term(str(e)) for e in range(8)],
            network_type=t5_type,
            concat_tensor_input=False,
        )
    networks = NetworkStore(select_switcher, tab_picker, col_picker)


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

def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--ss_type', type=str, choices=["t5", "bert"], default="t5", help='model used for select switcher')
    arg_parser.add_argument('--epoch', type=int, default=7, help='number of epoches')
    arg_parser.add_argument('--train_batch', type=int, default=8, help='batch size for training')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    args = arg_parser.parse_args()
    return args


def main():
    args = init()
    gc.collect()
    torch.cuda.empty_cache()
    run(
        epochs=args.epoch,
        lr=args.lr,
        seed=42,
        log_freq=50,
        train_batch_size=args.train_batch,
        val_batch_size=8,
        ss_type = args.ss_type,
        task = "task1"
        # verbose_building=False,
    )



if __name__ == "__main__":
    main()