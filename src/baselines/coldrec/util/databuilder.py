import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict


class DataBuilder(object):
    def __init__(self, training_data, valid_data, test_data):
        self.training_data = training_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.user = {}
        self.item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.valid_set_u = defaultdict(dict)
        self.valid_set_i = defaultdict(dict)
        self.test_set_u = defaultdict(dict)
        self.test_set_i = defaultdict(dict)
        self.__generate_set()
        self.user_num = len(self.user)
        self.item_num = len(self.item)

    def __generate_set(self):
        for entry in self.training_data:
            user, item, rating = entry[0], entry[1], entry[2]
            if user not in self.user:
                self.user[user] = len(self.user)
            if item not in self.item:
                self.item[item] = len(self.item)
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
        for entry in self.valid_data:
            user, item, rating = entry[0], entry[1], entry[2]
            if user not in self.user:
                self.user[user] = len(self.user)
            if item not in self.item:
                self.item[item] = len(self.item)
            self.valid_set_u[user][item] = rating
            self.valid_set_i[item][user] = rating
        for entry in self.test_data:
            user, item, rating = entry[0], entry[1], entry[2]
            if user not in self.user:
                self.user[user] = len(self.user)
            if item not in self.item:
                self.item[item] = len(self.item)
            self.test_set_u[user][item] = rating
            self.test_set_i[item][user] = rating

    def get_user_id(self, user):
        return self.user.get(user, -1)

    def get_item_id(self, item):
        return self.item.get(item, -1)

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def valid_size(self):
        return len(self.valid_set_u), len(self.valid_set_i), len(self.valid_data)

    def test_size(self):
        return len(self.test_set_u), len(self.test_set_i), len(self.test_data)

    def contain(self, u, i):
        return u in self.user and i in self.training_set_u.get(u, {})

    def contain_user(self, u):
        return u in self.user

    def contain_item(self, i):
        return i in self.item

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        k, v = self.user_rated(u)
        r = []
        for item in k:
            r.append(self.item[item])
        return r, v

    def col(self, i):
        k, v = self.item_rated(i)
        c = []
        for user in k:
            c.append(self.user[user])
        return c, v

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.training_set_u:
            for i in self.training_set_u[u]:
                m[self.user[u]][self.item[i]] = self.training_set_u[u][i]
        return m


class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(mat):
        mat = mat.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((mat.row, mat.col)).astype(np.int64)
        )
        values = torch.from_numpy(mat.data)
        shape = torch.Size(mat.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    @staticmethod
    def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
        from torch_sparse import SparseTensor, fill_diag, sum as sparsesum, mul
        row, col = edge_index
        adj = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(num_nodes, num_nodes))
        if normalization == 'sym':
            deg = sparsesum(adj, dim=1)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
            adj = mul(adj, deg_inv_sqrt.view(-1, 1))
            adj = mul(adj, deg_inv_sqrt.view(1, -1))
        elif normalization == 'rw':
            deg = sparsesum(adj, dim=1)
            deg_inv = 1.0 / deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
            adj = mul(adj, deg_inv.view(-1, 1))
        return adj

    @staticmethod
    def get_dense_laplacian(adj, normalization='none'):
        if normalization == 'sym':
            rowsum = torch.sum(adj, -1)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
            L = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        elif normalization == 'rw':
            rowsum = torch.sum(adj, -1)
            d_inv = torch.pow(rowsum, -1)
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv = torch.diagflat(d_inv)
            L = torch.mm(d_mat_inv, adj)
        else:
            L = adj
        return L


class ColdStartDataBuilder(object):
    def __init__(self, training_data, warm_valid_data, cold_valid_data, overall_valid_data,
                 warm_test_data, cold_test_data, overall_test_data,
                 user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx,
                 user_content=None, item_content=None):
        self.training_data = training_data
        self.warm_valid_data = warm_valid_data
        self.cold_valid_data = cold_valid_data
        self.overall_valid_data = overall_valid_data
        self.warm_test_data = warm_test_data
        self.cold_test_data = cold_test_data
        self.overall_test_data = overall_test_data

        self.user = {}
        self.item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set_u = defaultdict(dict)
        self.test_set_i = defaultdict(dict)
        self.valid_set_u = defaultdict(dict)
        self.valid_set_i = defaultdict(dict)

        self.warm_user_idx = warm_user_idx
        self.warm_item_idx = warm_item_idx
        self.cold_user_idx = cold_user_idx
        self.cold_item_idx = cold_item_idx

        self.source_user_content = {}
        self.source_item_content = {}

        self.generate_set()

        self.user_num = user_num
        self.item_num = item_num

        if user_content is not None:
            self._load_content(user_content, is_user=True)
        if item_content is not None:
            self._load_content(item_content, is_user=False)

        self.user_content = None
        self.item_content = None
        if user_content is not None:
            self.user_content = self._align_content(self.source_user_content, self.user, user_num)
        if item_content is not None:
            self.item_content = self._align_content(self.source_item_content, self.item, item_num)

    def generate_set(self):
        for entry in self.training_data:
            user, item, rating = entry[0], entry[1], entry[2]
            if user not in self.user:
                self.user[user] = len(self.user)
            if item not in self.item:
                self.item[item] = len(self.item)
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating

        for entry in (self.warm_valid_data + self.cold_valid_data + self.overall_valid_data):
            user, item, rating = entry[0], entry[1], entry[2]
            if user not in self.user:
                self.user[user] = len(self.user)
            if item not in self.item:
                self.item[item] = len(self.item)
            self.valid_set_u[user][item] = rating
            self.valid_set_i[item][user] = rating

        for entry in (self.warm_test_data + self.cold_test_data + self.overall_test_data):
            user, item, rating = entry[0], entry[1], entry[2]
            if user not in self.user:
                self.user[user] = len(self.user)
            if item not in self.item:
                self.item[item] = len(self.item)
            self.test_set_u[user][item] = rating
            self.test_set_i[item][user] = rating

    def _load_content(self, content_array, is_user):
        for src_id, feat in enumerate(content_array):
            if is_user:
                self.source_user_content[src_id] = feat
            else:
                self.source_item_content[src_id] = feat

    def _align_content(self, source_content, entity_map, num_entities):
        feat_dim = next(iter(source_content.values())).shape[0]
        out = np.zeros((num_entities, feat_dim), dtype=np.float32)
        for src_id, feat in source_content.items():
            dense_id = entity_map.get(src_id, None)
            if dense_id is not None and dense_id < num_entities:
                out[dense_id] = feat
        return out

    def get_user_id(self, user):
        return self.user.get(user, -1)

    def get_item_id(self, item):
        return self.item.get(item, -1)

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def contain(self, u, i):
        return u in self.user and i in self.training_set_u.get(u, {})

    def contain_user(self, u):
        return u in self.user

    def contain_item(self, i):
        return i in self.item

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def get_item_neighbors(self, item_list):
        return item_list

    def get_adj_mat(self):
        n_users = self.user_num
        n_items = self.item_num
        rows, cols, vals = [], [], []
        for u, items in self.training_set_u.items():
            uid = self.user[u]
            for item, r in items.items():
                iid = self.item[item]
                rows.append(uid)
                cols.append(iid)
                vals.append(r)
        R = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        return R

    def get_norm_adj_mat(self):
        n = self.user_num + self.item_num
        R = self.get_adj_mat()
        adj = sp.block_array([[None, R], [R.T, None]], format='csr')
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj).dot(d_mat)
        return norm_adj.tocsr()

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0] + adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix(
            (np.concatenate([ratings_keep, ratings_keep]),
             (np.concatenate([user_np_keep, item_np_keep + adj_shape[0]]),
              np.concatenate([item_np_keep + adj_shape[0], user_np_keep]))),
            shape=(n_nodes, n_nodes))
        return tmp_adj


class ColdStartMetaDataBuilder(ColdStartDataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
