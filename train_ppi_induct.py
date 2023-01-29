import hydra
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from tqdm import tqdm
from hydra import utils

import torch
from torch_scatter import scatter
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

from models import return_net
from layers import orthonomal_loss


def train(cfg, loader, model, optimizer, device):
    # train
    model.train()
    criteria = torch.nn.BCEWithLogitsLoss()

    for batch_id, data in enumerate(loader): # in [g1, g2, ..., g20]
        data = data.to(device)
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        loss = criteria(out, data.y)
        if cfg.skip_connection == 'summarize': # if it is proposal model
            loss += cfg.coef_orthonomal * orthonomal_loss(model, device)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(loader, model, device):
    model.eval()

    ys = {'valid': [], 'test': []}
    preds = {'valid': [], 'test': []}
    for mask in ['valid', 'test']:
        for data in loader[mask]: # only one batch (=g1+g2)
            data = data.to(device)
            ys[mask].append(data.y)
            _, out = model(data.x, data.edge_index)
            preds[mask].append((out > 0).float().cpu())

    y_valid    = torch.cat(ys['valid'], dim=0).to('cpu').detach().numpy().copy()
    pred_valid = torch.cat(preds['valid'], dim=0).to('cpu').detach().numpy().copy()
    y_test    = torch.cat(ys['test'], dim=0).to('cpu').detach().numpy().copy()
    pred_test = torch.cat(preds['test'], dim=0).to('cpu').detach().numpy().copy()
    return f1_score(y_valid, pred_valid, average='micro') if pred_valid.sum() > 0 else 0, \
           f1_score(y_test, pred_test, average='micro') if pred_test.sum() > 0 else 0


def train_and_test(cfg, data_loader, device):
    train_loader, val_loader, test_loader = data_loader

    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(cfg, train_loader, model, optimizer, device)

    loader = {'valid': val_loader, 'test': test_loader}
    return test(loader, model, device)


def run(cfg, root, device):
    train_dataset = PPI(root, split='train')
    val_dataset   = PPI(root, split='val')
    test_dataset  = PPI(root, split='test')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    data_loader = [train_loader, val_loader, test_loader]

    artifacts, valid_acces, test_acces = {}, [], []
    for tri in range(cfg['n_tri']):
        valid_acc, test_acc = train_and_test(cfg, data_loader, device)
        valid_acces.append(valid_acc)
        test_acces.append(test_acc)

    return artifacts, valid_acces, test_acces