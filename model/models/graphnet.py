import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from itertools import permutations
import scipy.sparse as sp

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
    
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx    


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    if torch.cuda.is_available():
        return torch.sparse.FloatTensor(indices, values, shape).cuda()
    else:
        return torch.sparse.FloatTensor(indices, values, shape)

class GraphFunc(nn.Module):
    def __init__(self, z_dim):
        super(GraphFunc, self).__init__()
        """
        DeepSets Function
        """
        self.gc1 = GraphConvolution(z_dim, z_dim * 4)
        self.gc2 = GraphConvolution(z_dim * 4, z_dim)
        self.z_dim = z_dim

    def forward(self, graph_input_raw, graph_label):
        """
        set_input, seq_length, set_size, dim
        """
        set_length, set_size, dim = graph_input_raw.shape
        assert(dim == self.z_dim)
        set_output_list = []
        
        for g_index in range(set_length):
            graph_input = graph_input_raw[g_index, :]
            # construct the adj matrix
            unique_class = np.unique(graph_label)
            edge_set = []
            for c in unique_class:
                current_index = np.where(graph_label == c)[0].tolist()
                if len(current_index) > 1:
                    edge_set.append(np.array(list(permutations(current_index, 2))))
            
            if len(edge_set) == 0:
                adj = sp.coo_matrix((np.array([0]), (np.array([0]), np.array([0]))),
                                    shape=(graph_label.shape[0], graph_label.shape[0]),
                                    dtype=np.float32)
            else:
                edge_set = np.concatenate(edge_set, 0)
                adj = sp.coo_matrix((np.ones(edge_set.shape[0]), (edge_set[:, 0], edge_set[:, 1])),
                                    shape=(graph_label.shape[0], graph_label.shape[0]),
                                    dtype=np.float32)        
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = normalize(adj + sp.eye(adj.shape[0]))
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            
            # do GCN process
            residual = graph_input
            graph_input = F.relu(self.gc1(graph_input, adj))
            graph_input = F.dropout(graph_input, 0.5, training=self.training)
            graph_input = self.gc2(graph_input, adj)        
            set_output = residual + graph_input
            set_output_list.append(set_output)
        
        return torch.stack(set_output_list)
    
class GCN(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = 640
        elif args.backbone_class == 'Res18':
            hdim = 512
        elif args.backbone_class == 'WRN':
            hdim = 640            
        else:
            raise ValueError('')
        
        self.graph_func = GraphFunc(hdim)
        
    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))
    
        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if self.training:
            graph_label = torch.arange(self.args.way).long()
        else:
            graph_label = torch.arange(self.args.eval_way).long()
        proto = self.graph_func(proto, graph_label)        
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)
        
        # do not use contrastive regularization for GCN (since there are only one sinlge instances class in each auxiliary task)
        if self.training:
            return logits, None            
        else:
            return logits   
