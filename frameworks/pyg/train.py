import argparse
import os
import random
import uuid

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

from monitor.monitor import Monitor


def fix_seed(seed):
    """
    Args :
        seed : value of the seed
    Function which allows to fix all the seed and get reproducible results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5)
        return x


def evaluate(model, g, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, batch in enumerate(dataloader):
        with torch.no_grad():
            out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            ys.append(y)
            y_hats.append(out)
    ys = torch.cat(ys)
    y_hats = torch.cat(y_hats)
    return sklearn.metrics.accuracy_score(ys.cpu().numpy(), y_hats.cpu().numpy())


def train(args, device, g, dataset, model, num_classes):
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    test_idx = dataset.test_idx.to(device)

    train_loader = NeighborLoader(
        g,
        input_nodes=train_idx,
        num_neighbors=[args.fanout] * args.n_layers,
        batch_size=args.bs,
        shuffle=True,
    )

    val_loader = NeighborLoader(
        g,
        input_nodes=val_idx,
        num_neighbors=[args.fanout] * args.n_layers,
        batch_size=args.bs,
        shuffle=True,
    )

    test_loader = NeighborLoader(
        g,
        input_nodes=test_idx,
        num_neighbors=[args.fanout] * args.n_layers,
        batch_size=args.bs,
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0
        for it, batch in enumerate(train_loader):
            opt.zero_grad()
            out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_loader, num_classes)
        print(
            "Epoch {:05d} | Loss {:.4f} | Val Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc
            )
        )

    acc = evaluate(model, g, test_loader, num_classes)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PygNodePropPredDataset(args.dataset, root='../../dataset')
    split_idx = dataset.get_idx_split()
    g = dataset[0]
    g = g.to(device)

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    model = SAGE(in_size, args.n_hidden, num_classes, args.n_layers).to(device)

    # model training
    print("Training...")
    train(args, device, g, dataset, model, num_classes)