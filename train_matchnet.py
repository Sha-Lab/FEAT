import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from feat.dataloader.samplers import CategoriesSampler
from feat.models.matchnet import MatchNet 
from feat.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, compute_confidence_interval
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--way', type=int, default=5)    
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_mul', type=float, default=1) # lr is the basic learning rate, while lr * lr_mul is the lr for other parts
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.2)    
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--use_bilstm', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='ConvNet', choices=['ConvNet', 'ResNet'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'CUB'])    
    parser.add_argument('--init_weights', type=str, default=None)    
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    save_path1 = '-'.join([args.dataset, args.model_type, 'MatchNet'])
    save_path2 = '_'.join([str(args.shot), str(args.query), str(args.way), 
                           str(args.step_size), str(args.gamma), str(args.lr), str(args.temperature)])
    if args.use_bilstm:
        args.save_path = save_path1 + '_' + save_path2 + '_' + str(args.lr_mul) + '_BiLSTM'
    else:
        args.save_path = save_path1 + '_' + save_path2
    ensure_path(args.save_path)

    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from feat.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from feat.dataloader.cub import CUB as Dataset
    else:
        raise ValueError('Non-supported Dataset.')
    
    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, 100, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, 500, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    
    model = MatchNet(args)
    if args.model_type == 'ConvNet':
        if args.use_bilstm:
            optimizer = torch.optim.Adam([{'params': model.encoder.parameters()},
                                         {'params': model.lstm.parameters(), 'lr': args.lr * args.lr_mul}], lr=args.lr)            
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model_type == 'ResNet':
        if args.use_bilstm:
            optimizer = torch.optim.SGD([{'params': model.encoder.parameters()},
                                         {'params': model.lstm.parameters(), 'lr': args.lr * args.lr_mul}], lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)                
        else:        
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)            
    else:
        raise ValueError('No Such Encoder')
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)        
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    
        # load pre-trained model (no FC weights)
        model_dict = model.state_dict()
        if args.init_weights is not None:
            pretrained_dict = torch.load(args.init_weights)['params']
            # remove weights for FC
            pretrained_dict = {k.replace('module', 'encoder'): v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    timer = Timer()
    global_count = 0
    writer = SummaryWriter(comment=args.save_path)
    
    label = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
        
    label_support = torch.arange(args.way).repeat(args.shot)
    label_support = label_support.type(torch.LongTensor)
    # transform to one-hot form
    label_support_onehot = torch.zeros(args.way * args.shot, args.way)
    label_support_onehot.scatter_(1, label_support.unsqueeze(1), 1)    
    if torch.cuda.is_available():
        label_support_onehot = label_support_onehot.cuda() # KN x N
            
    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()
        model.train()
        tl = Averager()
        ta = Averager()
            
        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p = args.shot * args.way
            data_shot, data_query = data[:p], data[p:]

            logits = model(data_shot, data_query) # KqN x KN x 1
            # use logits to weights all labels, KN x N
            prediction = torch.sum(torch.mul(logits, label_support_onehot.unsqueeze(0)), 1) # KqN x N
            # compute loss
            loss = F.cross_entropy(prediction, label)
            acc = count_acc(prediction, label)
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        label = torch.arange(args.way).repeat(args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
            
        print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = args.shot * args.way
                data_shot, data_query = data[:p], data[p:]
    
                logits = model(data_shot, data_query) # KqN x KN x 1
                # use logits to weights all labels, KN x N
                prediction = torch.sum(torch.mul(logits, label_support_onehot.unsqueeze(0)), 1) # KqN x N
                # compute loss
                loss = F.cross_entropy(prediction, label)
                acc = count_acc(prediction, label)
                vl.add(loss.item())
                va.add(acc)

        vl = vl.item()
        va = va.item()
        writer.add_scalar('data/val_loss', float(vl), global_val_count)
        writer.add_scalar('data/val_acc', float(va), global_val_count)    
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    writer.close()

    # Test Phase
    trlog = torch.load(osp.join(args.save_path, 'trlog'))
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, 10000, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    test_acc_record = np.zeros((10000,))

    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
    model.eval()

    ave_acc = Averager()
        
    for i, batch in enumerate(loader, 1):
        if torch.cuda.is_available():
            data, _ = [_.cuda() for _ in batch]
        else:
            data = batch[0]
        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]
        logits = model(data_shot, data_query) # KqN x KN x 1
        # use logits to weights all labels, KN x N
        prediction = torch.sum(torch.mul(logits, label_support_onehot.unsqueeze(0)), 1) # KqN x N
        acc = count_acc(prediction, label)
        ave_acc.add(acc)
        test_acc_record[i-1] = acc
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        
    
    m, pm = compute_confidence_interval(test_acc_record)
    print('Val Best Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc'], ave_acc.item()))
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))