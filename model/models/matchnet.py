import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel
from model.utils import one_hot

# Note: This is the MatchingNet without FCE
#       it predicts an instance based on nearest neighbor rule (not Nearest center mean)

class MatchNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))

        if self.training:
            label_support = torch.arange(self.args.way).repeat(self.args.shot).type(torch.LongTensor)
            label_support_onehot = one_hot(label_support, self.args.way)
        else:
            label_support = torch.arange(self.args.eval_way).repeat(self.args.eval_shot).type(torch.LongTensor)
            label_support_onehot = one_hot(label_support, self.args.eval_way)
        if torch.cuda.is_available():
            label_support_onehot = label_support_onehot.cuda() # KN x N

        # get mean of the support
        num_batch = support.shape[0]
        num_way = support.shape[2]
        num_support = np.prod(support.shape[1:3])
        num_query = np.prod(query_idx.shape[-2:])
        support = support.view(num_batch, num_support, emb_dim) # Ntask x NK x d
        label_support_onehot = label_support_onehot.unsqueeze(0).repeat(num_batch, 1, 1)
        
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)

        support = F.normalize(support, dim=-1) # normalize for cosine distance
        query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

        # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
        logits = torch.bmm(query, support.permute([0,2,1])) 
        logits = torch.bmm(logits, label_support_onehot) / self.args.temperature # KqN x N
        logits = logits.view(-1, num_way)

        if self.training:
            return logits, None
        else:
            return logits
