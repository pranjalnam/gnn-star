from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
import argparse
import os
import random
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def fix_seed(seed):
    '''
    Args :
        seed : value of the seed
    Function which allows to fix all the seed and get reproducible results
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    print(f"Training in {device} mode.")

    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root='../../dataset'))
    g = dataset[0]
    print("Dataset loaded into memory. Will shift to device in 10 seconds...")
    time.sleep(10)
    g = g.to(device)
    print(f"Dataset moved to {g.device}. Will move the features in 10 seconds...")
    time.sleep(10)
    x = g.ndata['feat']
    print(f"Dataset feat moved to {x.device}. Creating sampler in 10 seconds...")
    time.sleep(10)  
    print("Moving training indices to GPU...")
    train_idx = dataset.train_idx.to(device)
    
    print("Creating sampler...")
    sampler = MultiLayerFullNeighborSampler(3)
    
    print("Creating dataloader...")
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=32,
        shuffle=True,
        drop_last=False,
    )
    print("Dataloader created. Starting in 10 seconds...")
    time.sleep(10) 
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        print("Minibatch - ", it)
        pass
    print("Training done. Terminating in 10 seconds...")
    time.sleep(10)
