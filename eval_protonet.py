import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from feat.dataloader.samplers import CategoriesSampler
from feat.models.protonet import ProtoNet
from feat.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, compute_confidence_interval
from tensorboardX import SummaryWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--model_type', type=str, default='ConvNet', choices=['ConvNet', 'ResNet'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'CUB', 'TieredImageNet'])    
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    args.temperature = 1 # we set temperature = 1 during test since it does not influence the results
    pprint(vars(args))

    set_gpu(args.gpu)
    
    if args.dataset == 'MiniImageNet':
        from feat.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from feat.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from feat.dataloader.tiered_imagenet import tieredImageNet as Dataset       
    else:
        raise ValueError('Non-supported Dataset.')

    model = ProtoNet(args)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()    
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, 10000, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    test_acc_record = np.zeros((10000,))

    model.load_state_dict(torch.load(args.model_path)['params'])
    model.eval()

    ave_acc = Averager()
    label = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
        
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = args.way * args.shot
            data_shot, data_query = data[:k], data[k:]
    
            logits = model(data_shot, data_query)
            acc = count_acc(logits, label)
            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        
    m, pm = compute_confidence_interval(test_acc_record)
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))