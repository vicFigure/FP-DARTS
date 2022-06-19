import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import json
import csv

import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

sys.path.append('../')
import utils
import compare_genotypes
from model import NetworkCIFAR as Network

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='tmp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--base_dir',type=str,help='model dir')
parser.add_argument('--ckpt_path',type=str, default=None, help='trained model path')
args = parser.parse_args()

#args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
#utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
#if not os.path.exists(args.save):
#  os.makedirs(args.save)
#fh = logging.FileHandler(os.path.join(args.save, 'eval_log-doubleSepConv_16C.txt'))
#fh.setFormatter(logging.Formatter(log_format))
#logging.getLogger().addHandler(fh)


def infer(valid_queue, model, criterion):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, volatile=True).cuda()
      target = Variable(target, volatile=True).cuda()
  
      logits, _ = model(input)
      loss = criterion(logits, target)
  
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)
  
      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg



def test(base_dir, genotype_name, ckpt_path):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  if args.seed is None: args.seed = -1
  if args.seed < 0:
    args.seed = np.random.randint(low=0, high=10000)
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  # dataLoader
  if args.dataset == 'cifar10':
      train_transform, valid_transform = utils._data_transforms_cifar10(args)
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
      args.n_classes = 10
  elif args.dataset == 'cifar100':
      train_transform, valid_transform = utils._data_transforms_cifar100(args)
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
      args.n_classes = 100
  elif args.dataset == 'svhn':
      train_transform, valid_transform = utils._data_transforms_svhn(args)
      train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
      valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)
      args.n_classes = 10

#  train_transform, valid_transform = utils._data_transforms_cifar10(args)
#  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
#  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  genotype_path = os.path.join(base_dir, 'results_of_7q/genotype')
#  genotype_path = os.path.join(base_dir, 'results_of_7q/genotype_sal')
  genotype_file = os.path.join(genotype_path, '%s.txt'%genotype_name)
  tmp_dict = json.load(open(genotype_file,'r'))
  genotype = genotypes.Genotype(**tmp_dict)
  print(genotype)
 
  model = Network(args.init_channels, args.n_classes, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  model.drop_path_prob = 0.0
  model.load_state_dict(torch.load(ckpt_path), strict=False)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  valid_acc, valid_obj = infer(valid_queue, model, criterion)
  print(valid_acc, valid_obj)


if __name__ == '__main__':
#  main() 
#  arch_eval_iter(choose_type='fromfile')

#  base_dir = 'search-EXP-20200625-091629/'
  base_dir = args.base_dir
  tmp_dir = os.path.join(base_dir, 'hyperband')
  if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
  genotype_name = -1
  
  # deal with negative names
  genotype_path = os.path.join(base_dir, 'results_of_7q/genotype')
  files = os.listdir(genotype_path)
  max_name = 0
  for f in files:
    tmp = int(f.split('.')[0])
    if tmp > max_name: max_name = tmp
  if genotype_name < 0: genotype_name = max_name+1 + genotype_name

  # deal with default ckpt_dir
  if args.ckpt_path is None:
    ckpt_dir = os.path.join(base_dir, 'hyperband/hyperband1-600epochs')
    args.ckpt_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[-1])


  print("\n############")
  print("genotype names: ", genotype_name)
  print("load ckpt path ", args.ckpt_path)
  print("############\n")
  test(base_dir, genotype_name, args.ckpt_path)

