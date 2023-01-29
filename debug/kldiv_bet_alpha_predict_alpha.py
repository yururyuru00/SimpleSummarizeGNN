from matplotlib.axis import Axis
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from scipy.special import softmax

import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def calc_kldiv(a_vec, b_vec):
    return np.sum([a * np.log(a/b) for a, b in zip(a_vec, b_vec)])

def make_binomial(n, p=0.5):
    dist = np.array( [(math.factorial(n) / (math.factorial(i) * math.factorial(n-i))) * np.power(p, i) * np.power(1-p, n-i) \
                       for i in range(1, n+1)] )
    return dist * sum(dist)

def calc_khop_homophily(data, center_node, k_hop):
    subgraph = k_hop_subgraph(node_idx   = center_node,
                              num_hops   = k_hop,
                              edge_index = data.edge_index)
    sub_nodes, _, _, _ = subgraph
    num_neighbors = sub_nodes.size()[0]
    self_label = data.y[center_node]
    sub_node_labels = data.y[sub_nodes]
    num_same_labels = torch.where(sub_node_labels == self_label)[0].size()[0]
    return num_same_labels / num_neighbors


dataset_name = 'PubMed' # Cora or CiteSeer or PubMed
num_layer = 9
temparature= 1

# make alpha_true based on 100% class label
try:
    alpha_true = np.load('./alpha/true/{}.npy'.format(dataset_name))
except FileNotFoundError:
    dataset = Planetoid(root      = '../data/'+dataset_name,
                        name      = dataset_name,
                        transform = T.AddSelfLoops())
    data = dataset[0]
    alpha_true  = np.zeros((data.num_nodes, 9)) # 9 is max of number of layers
    homophilies = []
    for v in tqdm(range(data.num_nodes)):
        for l in range(9):
            homophily = calc_khop_homophily(data, v, k_hop=l+1)
            alpha_true[v][l] = homophily
    np.save('./alpha/true/{}'.format(dataset_name), alpha_true)

print(alpha_true, end='\n\n')
# normalize alpha_true
alpha_true = alpha_true[:, :num_layer]
alpha_true = softmax(alpha_true, axis=-1)
print(alpha_true)


# load alpha_predicted
alphas = {}
alphas['TwinGNN'] = np.load('./alpha/{}_alpha_L{}_t{}.npy'.format(dataset_name, num_layer, temparature))
n_nodes = alphas['TwinGNN'].shape[0]
alphas['GCN']  = np.repeat(np.identity(num_layer)[-1].reshape(1, -1), n_nodes, axis=0)


# calculate kl divergence between true and predicted alpha
epsilon_vec = np.array([1e-5 for _ in range(num_layer)])
kl_divs = {}
for model_name, alpha in alphas.items():
    kl_divs[model_name] = []
    for v_alpha_true, v_alpha in zip(alpha, alpha_true):
        v_alpha_true += epsilon_vec
        v_alpha += epsilon_vec
        kl_div = calc_kldiv(v_alpha_true, v_alpha)
        kl_divs[model_name].append(kl_div)

divs  = [v for n, v in kl_divs.items()]
names = [n for n, v in kl_divs.items()]

fig, ax = plt.subplots()
ax.boxplot(divs, sym='')
ax.set_xticklabels(names)
plt.tick_params(labelsize=18)
plt.ylim([0,3])
ax.set_yticks([0, 1, 2, 3])
plt.savefig('./visualize_alpha/{}/kldiv_gnn.png'.format(dataset_name))