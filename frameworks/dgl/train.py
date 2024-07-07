import argparse
import os
import random
import uuid
import warnings

import dgl
import dgl.nn as dglnn
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerNeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset

from monitor.monitor import Monitor

warnings.filterwarnings("ignore", category=UserWarning)


def fix_seed(seed):
    """
    Args :
        seed : value of the seed
    Function which allows to fix all the seed and get reproducible results
    """
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


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        for _ in range(n_layers - 2):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x).argmax(dim=1))
    ys = torch.cat(ys)
    y_hats = torch.cat(y_hats)
    return sklearn.metrics.accuracy_score(ys.cpu().numpy(), y_hats.cpu().numpy())


def train(args, device, g, dataset, model, num_classes):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    test_idx = dataset.test_idx.to(device)
    sampler = MultiLayerNeighborSampler(fanouts=[args.fanout] * args.n_layers)
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        batch_size=args.bs,
        shuffle=True,
        drop_last=False,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        batch_size=args.bs,
        shuffle=True,
        drop_last=False,
    )

    test_dataloader = DataLoader(
        g,
        test_idx,
        sampler,
        batch_size=args.bs,
        shuffle=False,
        drop_last=False,
    )

    opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            blocks = [b.to(device) for b in blocks]
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader, num_classes)
        print(
            "Epoch {:05d} | Loss {:.4f} | Val Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc
            )
        )

    acc = evaluate(model, g, test_dataloader, num_classes)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == "__main__":
    fix_seed(11)
    
    session_id = str(uuid.uuid4().hex)
    print(f"Session ID - {session_id}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--monitor", action='store_true')
    parser.add_argument("--seed", type=int, default=11)
   
    args = parser.parse_args()

    if args.monitor:
        # get process id
        pid = os.getpid()
        # launch monitoring script monitor.py in background
        Monitor(id=session_id, pid=pid)

    mode = "gpu"
    if not torch.cuda.is_available():
        mode = "cpu"
    print(f"Training in {mode} mode.")

    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root='../../dataset'))
    g = dataset[0]
    device = torch.device("cpu" if mode == "cpu" else "cuda")
    g = g.to("cuda" if mode == "gpu" else "cpu")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    model = SAGE(in_size, args.n_hidden, num_classes, args.n_layers).to(device)

    # model training
    print("Training...")
    train(args, device, g, dataset, model, num_classes)


