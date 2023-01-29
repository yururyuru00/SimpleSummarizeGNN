import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.datasets import KarateClub, Planetoid


def pre_transform(data): # for num_labels <= 3 (rgb)
    idx = torch.where(data.y == 0)
    data.y[idx] = 2
    idx = torch.where(data.y == 3)
    data.y[idx] = 0

def make_alpha_accum(data, alpha, sub_nodes_each_layer):
    max_depth = len(sub_nodes_each_layer)
    alpha_accum = [0. for _ in range(data.num_nodes)]
    for l in range(max_depth):
        for v in sub_nodes_each_layer[l]:
            alpha_accum[v] += alpha[l]
    return alpha_accum

def make_colors(data, alpha):
    one     = np.array( [255, 255, 255] )
    rgb_inv = np.array( [ [0  , 255, 255],
                          [255,   0, 255],
                          [255, 255,   0] ] )
    colors = []
    for v in range(data.num_nodes):
        label_v = data.y[v]
        rgb = one - alpha[v] * rgb_inv[label_v]
        colors.append( (int(rgb[0]),int(rgb[1]),int(rgb[2])) )
    return colors


def visualize_alpha(data, center_node, alpha, max_depth,
                    seed=42, partial_visualize=True, save_file='test.png'):
    # make subgraph of graph
    sub_nodes_each_layer = []
    for l in range(max_depth):
        subgraph = k_hop_subgraph(node_idx   = center_node,
                                  num_hops   = l+1,
                                  edge_index = data.edge_index)
        sub_nodes, sub_edge_index, _, _ = subgraph
        sub_nodes_each_layer.append(sub_nodes)

    # make color-maps based on each node of alpha
    alpha_accum = make_alpha_accum(data, alpha, sub_nodes_each_layer)
    node_colors = make_colors(data, alpha_accum)

    # visualize
    if partial_visualize:
        data = Data(data.x[sub_nodes], sub_edge_index, data.y[sub_nodes])
        target_nodes = sub_nodes.tolist()
    else: # if full visualize
        data = data
        target_nodes = range(data.num_nodes)
    center_node_color = node_colors[center_node]
    target_node_colors = [node_colors[v] for v in target_nodes]
    plt.figure(figsize=(10,10))
    G = to_networkx(data)
    # pos = nx.spring_layout(G, k=0.001, seed=seed)
    pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", root=center_node, args="")
    nx.draw_networkx_nodes(G, pos,
                              nodelist   = target_nodes,
                              edgecolors = 'black',
                              node_size  = 300,
                              node_color = [ '#%02x%02x%02x' % rgb
                                for rgb in target_node_colors ])
    nx.draw_networkx_nodes(G, pos,
                              nodelist   = [center_node],
                              edgecolors = 'black',
                              node_size  = 1500,
                              node_shape = '*',
                              node_color = '#%02x%02x%02x' % center_node_color)
    nx.draw_networkx_edges(G, pos, width=0.0, alpha=0.3, arrows=False)
    plt.savefig(save_file)



if __name__ == "__main__":
    dataset_name = 'PubMed' # only PubMed
    num_layer = 3
    temparature = 1

    # load data and pre_transform
    if dataset_name == 'KarateClub':
        dataset = KarateClub(transform = T.AddSelfLoops())
        pre_transform(dataset[0])
    else:
        dataset = Planetoid(root      = '../data/'+dataset_name,
                            name      = dataset_name,
                            transform = T.AddSelfLoops())
    data = dataset[0]

    # calculate accuracy of each node by gnn and twin-gnn
    corrects_twin = np.array([np.load('./correct_idxes/{}/twingnn/correct_{}.npy'.\
                                      format(dataset_name, i)) for i in range(10)])
    corrects_gnn = np.array([np.load('./correct_idxes/{}/gnn/correct_{}.npy'.\
                                      format(dataset_name, i)) for i in range(10)])
    ave_correct_twin = np.mean(corrects_twin, axis=0)
    ave_correct_gnn  = np.mean(corrects_gnn, axis=0)
    diff_correct = ave_correct_twin - ave_correct_gnn
    target_idxes = np.argsort(-1*diff_correct)

    test_mask = np.load('./correct_idxes/{}/test_mask.npy'.format(dataset_name))
    alpha = np.load('./alpha/{}_alpha_L{}_t{}.npy'.format(dataset_name, num_layer, temparature))
    num_layer = alpha.shape[-1]

    target_idxes_of_test = [t.item() for t in target_idxes if test_mask[t]]
    file = open('./visualize_alpha/{}/L3/memo.txt'.format(dataset_name), 'w')
    for rank, v in enumerate(target_idxes_of_test):
        if diff_correct[v] < 0.5:
            break

        file.write('rank{}(+{:.0f}%) node\'s alpha: {}\n'.format(rank+1, diff_correct[v]*100, alpha[v]))
        visualize_alpha(data, v, alpha[v], num_layer,
                        seed=42, partial_visualize=True,
                        save_file = './visualize_alpha/{}/L3/rank{}node_twingnn.png'.format(dataset_name, rank+1))
        visualize_alpha(data, v, np.array([0.,0.,0.,0.,1.]), num_layer,
                        seed=42, partial_visualize=True,
                        save_file = './visualize_alpha/{}/L3/rank{}node_gnn.png'.format(dataset_name, rank+1))
    file.close()