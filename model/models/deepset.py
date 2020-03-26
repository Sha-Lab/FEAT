import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel

class DeepSetsFunc(nn.Module):
    def __init__(self, z_dim):
        super(DeepSetsFunc, self).__init__()
        """
        DeepSets Function
        """
        self.gen1 = nn.Linear(z_dim, z_dim * 4)
        self.gen2 = nn.Linear(z_dim*4, z_dim)
        self.gen3 = nn.Linear(z_dim * 2, z_dim * 4)
        self.gen4 = nn.Linear(z_dim*4, z_dim)
        self.z_dim = z_dim

    def forward(self, set_input):
        """
        set_input, seq_length, set_size, dim
        """
        set_length, set_size, dim = set_input.shape
        assert(dim == self.z_dim)
        mask_one = torch.ones(set_size, set_size) - torch.eye(set_size, set_size)
        mask_one = mask_one.view(1, set_size, set_size, 1)
        if torch.cuda.is_available():
            mask_one = mask_one.cuda()
            
        combined_mean = torch.mul(set_input.unsqueeze(2), mask_one).max(1)[0] # 75 x 6 x 64, we can also try max here
        # do a bilinear transformation
        combined_mean = F.relu(self.gen1(combined_mean.view(-1, self.z_dim)))
        combined_mean = self.gen2(combined_mean)
        combined_mean_cat = torch.cat([set_input.contiguous().view(-1, self.z_dim), combined_mean], 1)
        # do linear transformation
        combined_mean_cat = F.relu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
    
        combined_mean_cat = combined_mean_cat.view(-1, set_size, self.z_dim)
        set_output = set_input + combined_mean_cat
        return set_output

    
class DeepSet(FewShotModel):
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
        
        self.set_func = DeepSetsFunc(hdim)      
        
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
        proto = self.set_func(proto)        
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
        
        # for regularization
        if self.training:
            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                  query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            # apply the transformation over the Aug Task
            aux_emb = self.set_func(aux_task) # T x N x (K+Kq) x d
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            
            if self.args.use_euclidean:
                aux_task = aux_task.contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
    
                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task.contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
    
                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)            
            
            return logits, logits_reg            
        else:
            return logits   
