import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from hydra import utils

import torch
from torch_scatter import scatter
from torch_geometric.loader import RandomNodeSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from models import return_net
from layers import orthonomal_loss


def train(cfg, loader, model, optimizer, device):
    # train
    model.train()
    criteria = torch.nn.BCEWithLogitsLoss()

    num_batches = len(loader)
    for batch_id, data in enumerate(loader): # in [g1, g2, ..., g20]
        data = data.to(device)
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        loss = criteria(out[data.train_mask], data.y[data.train_mask])
        if cfg.skip_connection == 'summarize': # if it is proposal model
            loss += cfg.coef_orthonomal * orthonomal_loss(model, device)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(loader, model, evaluator, device):
    model.eval()

    ys_valid, preds_valid = [], []
    ys_test, preds_test = [], []
    alphas, outs =[], []
    for data in loader: # only one graph (=g1+g2)
        data = data.to(device)
        alpha, out = model(data.x, data.edge_index)
        outs.append(out)
        alphas.append(alpha)
        ys_valid.append(data.y[data['valid_mask']].cpu())
        preds_valid.append(out[data['valid_mask']].cpu())
        ys_test.append(data.y[data['test_mask']].cpu())
        preds_test.append(out[data['test_mask']].cpu())

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(ys_valid, dim=0),
        'y_pred': torch.cat(preds_valid, dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(ys_test, dim=0),
        'y_pred': torch.cat(preds_test, dim=0),
    })['rocauc']

    return torch.cat(outs, axis=0), torch.cat(alphas, axis=0), valid_rocauc.item(), test_rocauc.item()


def train_and_test(cfg, data_loader):
    train_loader, test_loader = data_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])
    evaluator = Evaluator('ogbn-proteins')

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(cfg, train_loader, model, optimizer, device)

    return test(test_loader, model, evaluator, device)


def run(cfg, root, device):
    dataset = PygNodePropPredDataset('ogbn-proteins', root)
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]
    data.node_species = None
    data.y = data.y.to(torch.float)
    
    # Initialize features of nodes by aggregating edge features.
    row, col = data.edge_index
    data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')
    cfg.n_feat = cfg.e_feat

    # Set split indices to masks.
    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[splitted_idx[split]] = True
        data[f'{split}_mask'] = mask

    train_loader = RandomNodeSampler(data, num_parts=40, shuffle=True,
                                     num_workers=0)
    test_loader = RandomNodeSampler(data, num_parts=5, num_workers=0)
    data_loader = [train_loader, test_loader]

    artifacts, valid_acces, test_acces = {}, [], []
    for tri in range(cfg['n_tri']):
        h, alpha, valid_acc, test_acc = train_and_test(cfg, data_loader)
        artifacts["y_true.npy"] = data.y
        artifacts[f"alpha_{tri}_test.npy"] = alpha
        artifacts[f"h_{tri}_test.npy"] = h
        valid_acces.append(valid_acc)
        test_acces.append(test_acc)

    return artifacts, valid_acces, test_acces