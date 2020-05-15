import torch
import scipy.sparse as sp
import numpy as np


class DataStruct:

    def __init__(self, sentence, adj, trigger, trigger_index, eep, index,sentence_emb=None):
        self.sentence = sentence
        self.adj = adj
        self.trigger = trigger
        self.trigger_index = trigger_index
        self.eep = eep
        self.index = index
        self.sentence_emb=sentence_emb

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)  # 生成对角矩阵
        mx = r_mat_inv.dot(mx)
        return mx

    def trans_data(self, adj_length):
        sp_adj = torch.tensor(self.adj).to_sparse()
        adj = sp.coo_matrix(
            (sp_adj.coalesce().values().numpy(), (sp_adj.coalesce().indices().numpy()[0], sp_adj.coalesce().indices().numpy()[1])),
            shape=(adj_length, adj_length),
            dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        return adj

    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def get_eep(self):
        return self.eep
