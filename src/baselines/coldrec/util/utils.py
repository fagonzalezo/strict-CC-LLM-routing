import random
import numpy as np
import torch
import torch.nn as nn
import sys


def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def next_batch_pairwise(data, batch_size):
    training_data = data.training_data
    shuffle_indices = np.random.permutation(np.arange(len(training_data)))
    training_data = [training_data[i] for i in shuffle_indices]
    ptr = 0
    data_size = len(training_data)
    num_batches = int(np.ceil(data_size / batch_size))
    for i in range(num_batches):
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(items[i])
            u_idx.append(user)
            neg_item = np.random.randint(len(item_list))
            while item_list[neg_item] in data.training_set_u[user]:
                neg_item = np.random.randint(len(item_list))
            j_idx.append(item_list[neg_item])
        yield u_idx, i_idx, j_idx


def next_batch_pairwise_with_neighbor(data, batch_size):
    training_data = data.training_data
    shuffle_indices = np.random.permutation(np.arange(len(training_data)))
    training_data = [training_data[i] for i in shuffle_indices]
    ptr = 0
    data_size = len(training_data)
    num_batches = int(np.ceil(data_size / batch_size))
    for i in range(num_batches):
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(items[i])
            u_idx.append(user)
            neg_item = np.random.randint(len(item_list))
            while item_list[neg_item] in data.training_set_u[user]:
                neg_item = np.random.randint(len(item_list))
            j_idx.append(item_list[neg_item])
        yield u_idx, i_idx, j_idx, data.get_item_neighbors(i_idx), data.get_item_neighbors(j_idx)


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)


def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)
    return emb_loss * reg


def mse_loss(output, target):
    return torch.mean((output - target) ** 2)


def InfoNCE(view1, view2, temperature, b_cos=True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score + 10e-6)
    return torch.mean(cl_loss)


def process_bar(num, total):
    rate = float(num) / total
    ratenum = int(50 * rate)
    r = '\r[{}{}]{}/{}'.format(
        '*' * ratenum, ' ' * (50 - ratenum), num, total)
    sys.stdout.write(r)
    sys.stdout.flush()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, activation='relu'):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
