import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import torch.utils.data.sampler
import numpy as np
import argparse
import tqdm
import torchnet as tnt
import collections

from utils import util
import random
cudnn.benchmark = True

seed=12
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--dset', default='opra')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')

parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--cv_dir', default='cv/tmp/',help='Directory for saving checkpoint models')
parser.add_argument('--save_every', default=0.5, type=float, help='fraction of an epoch to save after')
parser.add_argument('--load', default=None)
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--max_len', default=3, type=int)
parser.add_argument('--subsample', default=1, type=int)
parser.add_argument('--max_iter', default=2300, type=int)
parser.add_argument('--parallel', action ='store_true', default=True)
parser.add_argument('--workers', type=int, default=0)
args = parser.parse_args()


os.makedirs(args.cv_dir, exist_ok=True)

def save(epoch, iteration,cv_dir):
    os.makedirs(cv_dir, exist_ok=True)
    print('Saving state, iter:', iteration)
    state_dict = net.state_dict() if not args.parallel else net.module.state_dict()
    checkpoint = {'net': state_dict, 'args': args, 'iter': iteration}

    torch.save(checkpoint, '%s/ckpt_E_%d_I_%d.pth' % (cv_dir, epoch, iteration))

def train(cv_dir,iteration=0):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers,
                                              sampler=trainset.data_sampler(),drop_last=True)
    net.train()
    total_iters = len(trainloader)
    epoch = iteration // total_iters
    save_every = int(args.save_every * len(trainloader))
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))
  
    while iteration <= args.max_iter:

        for batch in trainloader:
            batch = util.batch_cuda(batch)

            pred, loss_dict = net(batch)

            loss_dict = {k: v.mean() for k, v in loss_dict.items()}
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred_idx = pred.max(1)
            correct = (pred_idx == batch['verb']).float().sum()
            batch_acc = correct / pred.shape[0]
            loss_meters['bacc %'].add(batch_acc.item())

            for k, v in loss_dict.items():
                loss_meters[k].add(v.item())
            loss_meters['total_loss'].add(loss.item())

            if iteration % args.print_every == 0:
                log_str = 'iter: %d (%d + %d/%d) | ' % (iteration, epoch, iteration % total_iters, total_iters)
                log_str += ' | '.join(['%s: %.3f' % (k, v.value()[0]) for k, v in loss_meters.items()])
                print(log_str)

            if iteration > 500  and iteration % save_every == 0:
                save(epoch, iteration, cv_dir)

            iteration += 1

        epoch += 1
        save(epoch, iteration,cv_dir)
       
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                                                  shuffle=False, num_workers=args.workers, 
                                                  sampler=trainset.data_sampler(),drop_last=True)

#----------------------------------------------------------------------------------------------------------------------------------------#

import data
from data import opra

trainset = opra.OPRAInteractions(root=data._DATA_ROOTS['opra'], split='train', 
                                         max_len=args.max_len,ratio=0.1,i_ratio=0.2)
recon = 'mse'

from models import model, backbones

torch.backends.cudnn.enabled = False 
net = model.hand_net(len(trainset.verbs), trainset.max_len,
                        backbone=backbones.dr50_n28,
                        ant_loss=recon,groups=2)
net.cuda()

start_iter = 0
if args.load:
    checkpoint = torch.load(args.load, map_location='cpu')
    weights,start_iter = checkpoint['net'], checkpoint['iter']
    net.load_state_dict(weights, strict=False)
    print ('Loaded checkpoint from %s'%os.path.basename(args.load))

if args.parallel:
    net = nn.DataParallel(net)

optim_params = list(filter(lambda p: p.requires_grad,net.parameters()))
print ('# params to optimize', len(optim_params))

optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.weight_decay)
cv_dir="save_models/"
train(cv_dir=cv_dir,iteration=0)


