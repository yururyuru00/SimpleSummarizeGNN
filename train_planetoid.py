import hydra
from hydra import utils
from tqdm import tqdm
from omegaconf import DictConfig
import mlflow

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from models import return_net
from layers import orthonomal_loss
from debug.gpu import dump_gpu_properties

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def train(cfg, data, model, optimizer, device):
    # train
    model.train()
    optimizer.zero_grad()

    # train by class label
    _, h = model(data.x, data.edge_index)
    prob_labels = F.log_softmax(h, dim=1)
    loss_train  = F.nll_loss(prob_labels[data.train_mask], data.y[data.train_mask])
    if cfg.skip_connection == 'summarize': # if it is proposal model
        loss_train += cfg.coef_orthonomal * orthonomal_loss(model, device)
    
    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    _, h = model(data.x, data.edge_index)
    prob_labels_val = F.log_softmax(h, dim=1)
    loss_val = F.nll_loss(prob_labels_val[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(prob_labels_val[data.val_mask], data.y[data.val_mask])
    
    return loss_val.item(), acc_val


def test(data, model):
    model.eval()
    alpha, h = model(data.x, data.edge_index)
    prob_labels_test = F.log_softmax(h, dim=1)
    acc = accuracy(prob_labels_test[data.test_mask], data.y[data.test_mask])

    return h, alpha, acc


def train_and_test(cfg, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])

    best_loss = 100.
    bad_counter = 0
    for epoch in range(1, cfg['epochs']):
        loss_val, acc_val = train(cfg, data, model, optimizer, device)

        if loss_val < best_loss:
            best_loss = loss_val
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == cfg['patience']:
            break

    h, alpha, acc_test = test(data, model)
    return h, alpha, acc_val, acc_test


def run(cfg, root, device):
    dataset = Planetoid(root          = root,
                        name          = cfg['dataset'],
                        split         = cfg['split'],
                        transform     = eval(cfg['transform']),
                        pre_transform = eval(cfg['pre_transform']))
    data = dataset[0].to(device)

    artifacts, acces_val, acces_test = {}, [], []
    for tri in tqdm(range(cfg['n_tri'])):
        h, alpha, acc_val, acc_test = train_and_test(cfg, data)
        artifacts["y_true.npy"] = data.y
        artifacts[f"alpha_{tri}_test.npy"] = alpha
        artifacts[f"h_{tri}_test.npy"] = h
        acces_val.append(acc_val.to('cpu').item())
        acces_test.append(acc_test.to('cpu').item())

    return artifacts, acces_val, acces_test

