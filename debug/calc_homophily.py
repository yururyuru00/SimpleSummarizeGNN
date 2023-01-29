import argparse
import torch
from torch_geometric.datasets import Planetoid, Reddit, WebKB
from torch_geometric.utils import homophily, k_hop_subgraph
from ogb.nodeproppred import PygNodePropPredDataset



parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='Cora')
parser.add_argument('--method', type=str, default='node', help='node or edge')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if args.name in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root      = '../data/' + args.name,
                        name      = args.name)
    data = dataset[0].to(device)

elif args.name in ['Cornell', 'Texas', 'Wisconsin']:
    dataset = WebKB(root      = '../data/' + args.name,
                    name      = args.name)
    data = dataset.data.to(device)

elif args.name == 'Reddit':
    dataset = Reddit(root = '../data/Reddit')
    data = dataset[0].to(device)

elif args.name == 'Arxiv':
    dataset = PygNodePropPredDataset('ogbn-arxiv', 
                                     '../data/ogbn-arxiv')
    splitted_idx = dataset.get_idx_split()
    data = dataset[0].to(device)
    data.node_species = None
    data.y = data.y.to(torch.float)


h = homophily(data.edge_index, data.y, method=args.method)
print('{}\'s {} homophily: {:.4f}'.format(args.name, args.method, h))