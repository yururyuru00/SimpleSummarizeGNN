import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from hydra import utils

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from models import return_net
from layers import orthonomal_loss


def train(cfg, data, model, optimizer, device):
    model.train()

    optimizer.zero_grad()
    _, out = model(data.x, data.adj_t)
    out = out.log_softmax(dim=-1)
    out = out[data['train_mask']]
    loss = F.nll_loss(out, data.y.squeeze(1)[data['train_mask']])
    if cfg.skip_connection == 'summarize': # if it is proposal model
        loss += cfg.coef_orthonomal * orthonomal_loss(model, device)
    
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data, model, evaluator):
    model.eval()
    alpha, out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)
    
    valid_acc = evaluator.eval({'y_true': data.y[data['valid_mask']],
                                'y_pred': y_pred[data['valid_mask']],})['acc']
    test_acc = evaluator.eval({'y_true': data.y[data['test_mask']],
                               'y_pred': y_pred[data['test_mask']],})['acc']
    
    return out, alpha, valid_acc, test_acc


def train_and_test(cfg, data, device):
    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])
    evaluator = Evaluator('ogbn-arxiv')

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(cfg, data, model, optimizer, device)

    return test(data, model, evaluator)


def run(cfg, root, device):
    dataset = PygNodePropPredDataset('ogbn-arxiv', root, transform=T.ToSparseTensor())
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    # Set split indices to masks.
    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[splitted_idx[split]] = True
        data[f'{split}_mask'] = mask

    artifacts, valid_acces, test_acces = {}, [], []
    for tri in range(cfg['n_tri']):
        h, alpha, valid_acc, test_acc = train_and_test(cfg, data, device)
        artifacts["y_true.npy"] = data.y
        artifacts[f"alpha_{tri}_test.npy"] = alpha
        artifacts[f"h_{tri}_test.npy"] = h
        valid_acces.append(valid_acc)
        test_acces.append(test_acc)

    return artifacts, valid_acces, test_acces