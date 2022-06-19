import os
import sys
import time
import itertools
import glob
import numpy as np
import json
import csv

import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
import model_search
from architect import Architect
from spaces import spaces_dict

import sys
sys.path.append('../')
import utils
import saliency_utils


#torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--space', type=str, default='s1', help='space index')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# save path
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--save_one_shot', action='store_true', default=False, help='whether save the one-shot model')

# For Prune
parser.add_argument('--sal_type', type=str, default='task', help='type of saliency: task or naive')
parser.add_argument('--warmup', type=int, default=10, help='epochs of warm up before pruning')
parser.add_argument('--num_compare', type=int, default=1, help='The number of candidate ops before pruning (The top least important in saliency). We need to calculate the loss and acc of these candidates and prune the least important one')
parser.add_argument('--iter_compare', type=int, default=30, help='The iteration (batch) to run each comparison candidate')

# For regularization of pruned alphas
parser.add_argument('--reg_type', type=str, default='norm', help='type of regularization: norm, gini, or entropy')
parser.add_argument('--reg_alpha', type=float, default=0.1, help='weight of KD_logits_Loss')
parser.add_argument('--sal_second', action='store_true', default=False, help='keep second order of Taylor Expansion')
parser.add_argument('--no_restrict', action='store_true', default=False, help='use cutout')

parser.add_argument('--task', type=int, default=0, help='task ID')


args = parser.parse_args()

args.save = 'ckpt/search-{}-{}-task{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), args.task)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


torch.backends.cudnn.deterministic = True


def main(primitives):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  
  if args.seed is None: args.seed = -1
  if args.seed < 0:
    args.seed = np.random.randint(low=0, high=10000)
  logging.info('seed = %d'%args.seed)
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
      args.n_classes = 10
  elif args.dataset == 'cifar100':
      train_transform, valid_transform = utils._data_transforms_cifar100(args)
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
      args.n_classes = 100
  elif args.dataset == 'svhn':
      train_transform, valid_transform = utils._data_transforms_svhn(args)
      train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
      args.n_classes = 10

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=0)

  sal_queue = torch.utils.data.DataLoader(
      train_data, batch_size=96 if args.sal_second else args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=0)

  
  # Construct Network
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = model_search.Network(args.init_channels, args.n_classes, args.layers, criterion, primitives)
  model = model.cuda()
  logging.info("model param size = %fMB", utils.count_parameters_in_MB(model))
  
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  val_loss = [] # record the loss of val for each epoch  - q1
  val_acc = [] # recod the acc of val for each epoch  - q2
  results = {}
  results['val_loss'] = []
  results['val_acc'] = []

  ckpt_dir = os.path.join(args.save, 'ckpt')
  result_dir = os.path.join(args.save, 'results_of_7q') # preserve the results
  genotype_dir = os.path.join(result_dir, 'genotype') # preserve the argmax genotype for each epoch  - q3,5,7
  genotype_sal_dir = os.path.join(result_dir, 'genotype_sal') # preserve the argmax genotype for each epoch  - q3,5,7
  alpha_dir = os.path.join(result_dir, 'alpha')
  sal_dir = os.path.join(result_dir, 'sal')
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
  if not os.path.exists(genotype_dir):
    os.makedirs(genotype_dir)
  if not os.path.exists(genotype_sal_dir):
    os.makedirs(genotype_sal_dir)
  if not os.path.exists(alpha_dir):
    os.makedirs(alpha_dir)
  if not os.path.exists(sal_dir):
    os.makedirs(sal_dir)


  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    if args.drop_path_prob != 0:
      model.drop_path_prob = args.drop_path_prob * epoch / (args.epochs - 1)
      train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
      logging.info('epoch %d lr %e drop_prob %e cutout_prob %e', epoch, lr,
                    model.drop_path_prob,
                    train_transform.transforms[-1].cutout_prob)
    else:
      logging.info('epoch %d lr %e', epoch, lr)

    model.update_softmax_arch_parameters()
    print(model.softmax_alphas_normal)
    print(model.softmax_alphas_reduce)

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, sal_dir, sal_queue)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)
 

    genotype = model.genotype(args.no_restrict)
    logging.info('genotype = %s', genotype)

    if args.save_one_shot:
      utils.save_supernet(model, os.path.join(ckpt_dir, 'weights_%d.pt'%epoch))

    # for seven questions
    #q1 & q2
    results['val_loss'].append(valid_obj)
    results['val_acc'].append(valid_acc)
    #q3,5,7
    genotype_file = os.path.join(genotype_dir, '%d.txt'%epoch)
    with open(genotype_file, 'w') as f:
      json.dump(genotype._asdict(), f)
      # to recover: genotype = genotype(**dict)
    #q6: save the alpha weights
    alpha_file = os.path.join(alpha_dir, '%d.txt'%epoch)
    alpha_weights = model.arch_parameters()
    alphas = {}
    alphas['alphas_normal'] = model.softmax_alphas_normal.data.cpu().numpy().tolist()
    alphas['alphas_reduce'] = model.softmax_alphas_reduce.data.cpu().numpy().tolist()
    with open(alpha_file, 'w') as f:
      json.dump(alphas, f)

  # save the results:
  result_file = os.path.join(result_dir, 'results.csv')
  with open(result_file, 'w') as f:
    writer = csv.writer(f)
    title = ['epoch', 'val_loss', 'val_acc']
    writer.writerow(title)
    for epoch, val_loss in enumerate(results['val_loss']):
      a = [epoch, val_loss, results['val_acc'][epoch]]
      writer.writerow(a)



def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, sal_dir, sal_queue):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  interval_prune = int(len(train_queue) / 2.)

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)  #input.size :[32,3,32,32]

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()

    model.update_softmax_arch_parameters()
    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
    model.update_softmax_arch_parameters()

    optimizer.zero_grad()
    logits= model(input) 

    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    # compute saliency & prune
    prune = epoch >= args.warmup and model.has_redundent_op(no_restrict=args.no_restrict)
#    prune = False
    if prune and (step+1)%interval_prune==0:
      optim_e = 0
      sal_file = os.path.join(sal_dir, '%d.txt'%epoch)
      torch.cuda.empty_cache()
      saliency = compute_sal(sal_queue, model, criterion, sal_type=args.sal_type, sal_file=sal_file, second_order=args.sal_second)
      torch.cuda.empty_cache()

      prune_masks = prune_by_mask(model, saliency, args.num_compare, optim_e, train_queue, valid_queue, no_restrict=args.no_restrict)
      model.update_softmax_arch_parameters()
      print("-"*10)
      print("prune_masks:")
      print(prune_masks[0])
      print(prune_masks[1])
      print("-"*10)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, early_stop=-1, verbose=True):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = Variable(input, volatile=True).cuda()
      target = Variable(target, volatile=True).cuda()

      logits = model(input)
      loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if verbose and step % args.report_freq == 0:
      logging.info('valid %03d |CELoss= %.3e |Acc= %.3f@1 %.3f@5', step, objs.avg, top1.avg, top5.avg)
    if early_stop > 0 and step >= early_stop: break
  return top1.avg, objs.avg


def compute_sal(valid_queue, model, criterion, sal_type='task', sal_file=None, second_order=False):
  ###################
  # naive saliency
  ###################
  if sal_type == 'naive':
    n_saliencys = model.get_softmax_arch_parameters()
    print("Saliency")
    print(n_saliencys)
    return n_saliencys

#  model.eval()
  model.train()
  if second_order:
      non_zero = 0.
      for mask in model.prune_masks:
        non_zero += mask.sum()
      early_stop = 500/non_zero
      print(non_zero, early_stop)
  else: early_stop = len(valid_queue)

  t_saliencys = [0, 0]
  l_saliencys = [0, 0]
  for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, volatile=True).cuda()
      target = Variable(target, volatile=True).cuda()


      logits = model(input)

      alphas = model.get_softmax_arch_parameters()
#      alphas = model.arch_parameters()

      ###################
      # task saliency
      ###################
#      model.build_softmax_alpha_hook()
      loss = criterion(logits, target)
#      loss.backward()
#      model.remove_softmax_alpha_hook()

      if second_order:
        saliency_utils.zero_grads(model.parameters())
        saliency_utils.zero_grads(alphas)
        hessian = saliency_utils._hessian(loss, alphas, model.prune_masks)
#        print(hessian)
        diags = hessian.diag()
        diags = [diags[:alphas[0].numel()].view(alphas[0].shape), diags[alphas[0].numel():].view(alphas[1].shape)]

        grads_alphas = torch.autograd.grad(loss, alphas, grad_outputs=None,
                                      allow_unused=True,
                                      retain_graph=None,
                                      create_graph=False)
        t_saliencys = [t_saliencys[k]+grad*(-alpha.detach())+diag*alpha.detach().pow(2) for k, (grad, alpha, diag) in enumerate(zip(grads_alphas, alphas, diags))]
        
      else:
        grads_alphas = torch.autograd.grad(loss, alphas, grad_outputs=None,
                                      allow_unused=True,
                                      retain_graph=None,
                                      create_graph=False)
        t_saliencys = [t_saliencys[k]+grad*(-alpha.detach()) for k, (grad, alpha) in enumerate(zip(grads_alphas, alphas))]
        
#        print(grads_alphas)
#        # judge whether grads_alphas is correct
#        grads_arch_param = torch.autograd.grad(loss, model.arch_parameters(), grad_outputs=None,
#                                      allow_unused=True,
#                                      retain_graph=None,
#                                      create_graph=False)
#        correct_alphas = []
#        for i, g in enumerate(grads_alphas):
#          const = (g*alphas[i]).sum(dim=-1, keepdim=True)
#          compute_grad_arch_param = (g-const) * alphas[i]
#          diff = (compute_grad_arch_param - grads_arch_param[i]).abs()
#          print(diff.min(), diff.max(), diff.sum())
#        assert(0)

      if step % args.report_freq == 0:
        logging.info('saliency step %03d/%d', step, early_stop)
      if early_stop > 0 and step >= early_stop: break


  saliency = [v/step for v in t_saliencys]
  print("Saliency")
  print(saliency)
  # save saliency
  if sal_file is not None:
    save_sal = {}
    save_sal['normal'] = saliency[0].data.cpu().numpy().tolist()
    save_sal['reduce'] = saliency[1].data.cpu().numpy().tolist()
    with open(sal_file, 'w') as f:
      json.dump(save_sal, f)
  return saliency
   

def prune_by_mask(model, saliency, num_compare=1, optim_e=0, train_queue=None, valid_queue=None, no_restrict=False):

  def can_prune_pos(mask, pos_r, pos_c):
    if mask[pos_r, pos_c] == 0: return False
    if no_restrict: return True
    start = 0
    n = 2
    for i in range(model._steps):
      end = start + n
      if start <= pos_r and end > pos_r:
        col_sum = mask[start:end].sum(dim=1)
        num_valid_row = (col_sum>0).sum()
        if num_valid_row > 2: return True
        elif num_valid_row == 2:
          if mask[pos_r].sum() > 1: return True
          return False
        else:
          raise(ValueError("The code runs wrong, since it cannot make sure each node wons at least 2 input edges"))
      start = end
      n += 1
    raise(ValueError("The code runs wrong, since there is no %d rows in the mask"%pos_r))

  def refine_sal(mask, sal, max_value):
    ''' Since DARTS add some restriction on the degree of each node, so if a node has and only has two edges, we cannot prune the edges to this node anymore'''
    col_sum = mask.sum(dim=1)
    num_valid_row = (col_sum>0).sum()
    if num_valid_row > 2: return sal
    elif num_valid_row == 2:
      for r in range(mask.shape[0]):
        if col_sum[r] == 0: continue
        elif col_sum[r] == 1: sal[r].fill_(max_value)
    else:
      raise(ValueError("The code runs wrong, since it cannot make sure each node wons at least 2 input edges"))
    return sal

  R,C = saliency[0].shape
  mask_normal, mask_reduce = model.prune_masks
  s_normal = saliency[0].abs()
  s_reduce = saliency[1].abs()
  max_normal = s_normal.max()
  max_reduce = s_reduce.max()
  # do not prune the same edge repeatedly
  s_normal = torch.where(mask_normal==1, s_normal, max_normal)
  s_reduce = torch.where(mask_reduce==1, s_reduce, max_reduce)
  if not no_restrict:
    # make sure each node owns at least 2 input edges
    n = 2
    start = 0
    for i in range(model._steps):
      end = start + n
      s_normal[start:end] = refine_sal(mask_normal[start:end], s_normal[start:end], max_normal)
      s_reduce[start:end] = refine_sal(mask_reduce[start:end], s_reduce[start:end], max_reduce)
  #    tmp = mask_normal[start:end]
  #    if tmp.sum() <= 2: s_normal[start:end].fill_(max_normal)
  #    tmp = mask_reduce[start:end]
  #    if tmp.sum() <= 2: s_reduce[start:end].fill_(max_reduce)
      start = end
      n += 1

  alpha_normal, alpha_reduce = model.get_softmax_arch_parameters()
  if num_compare <= 1:
      # update mask_normal
#      tmp, pos_c = s_normal.min(dim=1)
#      pos_r = tmp.argmin(dim=0)
#      pos_c = pos_c[pos_r]
      s_normal = s_normal.view(-1).cpu().data.numpy()
      idx = s_normal.argsort()[0]
      pos_r = int(idx / C)
      pos_c = int(idx - pos_r*C)

      mask_normal[pos_r, pos_c] = 0
      sal_change = s_normal[idx]
      print("Prune normal idx: ", pos_r, pos_c)
      print("Saliency and beta: ", s_normal[idx], alpha_normal[pos_r, pos_c])

      # update mask_reduce 
#      tmp, pos_c = s_reduce.min(dim=1)
#      pos_r = tmp.argmin(dim=0)
#      pos_c = pos_c[pos_r]
      s_reduce = s_reduce.view(-1).cpu().data.numpy()
      idx = s_reduce.argsort()[0]
      pos_r = int(idx / C)
      pos_c = int(idx - pos_r*C)

      mask_reduce[pos_r, pos_c] = 0
      sal_change += s_reduce[idx]
      print("Prune reduce idx: ", pos_r, pos_c)
      print("Saliency and beta: ", s_reduce[idx], alpha_reduce[pos_r, pos_c])
  else:
      new_model = model.new()
      model_dict = model.state_dict()
      new_model.load_state_dict(model_dict, strict=True)

      new_model.update_softmax_arch_parameters()
      new_alpha_normal, new_alpha_reduce = new_model.get_softmax_arch_parameters()
      new_alpha_normal.data.mul_(mask_normal.float())
      new_alpha_reduce.data.mul_(mask_reduce.float())

      s_normal = s_normal.view(-1).cpu().data.numpy()
      idxes = s_normal.argsort()[:num_compare]
#      print('debug:', s_normal[idxes])
      best_acc = 0.
      best_loss = 100.
      accs = []
      losses = []
      for idx in idxes:
        pos_r = int(idx / C)
        pos_c = int(idx - pos_r*C)
        if not can_prune_pos(mask_normal, pos_r, pos_c): continue
#        if mask_normal[pos_r, pos_c] == 0: continue
#        print('debug:', pos_r, pos_c, s_normal[idx], saliency[0].abs()[pos_r, pos_c])
        tmp_alpha = new_alpha_normal[pos_r, pos_c].item()
        new_alpha_normal[pos_r, pos_c].fill_(0.)
        criterion = nn.CrossEntropyLoss()
        acc, loss = infer(valid_queue, new_model, criterion, early_stop=args.iter_compare, verbose=False)
        del criterion
#        print('debug:', new_alpha_normal)
        accs.append(acc)
        losses.append(loss)
#        if acc > best_acc:
        if loss < best_loss:
          best_pos_r = pos_r
          best_pos_c = pos_c
          best_acc = acc
          best_loss = loss
        new_alpha_normal[pos_r, pos_c].fill_(tmp_alpha)
#        print('debug:', new_alpha_normal)
      mask_normal[best_pos_r, best_pos_c] = 0
      sal_change = s_normal[best_pos_r*C+best_pos_c]
      print("Prune normal idx: ", best_pos_r, best_pos_c)
      print('accs', accs)
      print('losses', losses)
      print("Saliency and beta and best acc & loss: ", s_normal[best_pos_r*C+best_pos_c], new_alpha_normal[best_pos_r, best_pos_c], best_acc, best_loss)
      s_reduce = s_reduce.view(-1).cpu().data.numpy()
      idxes = s_reduce.argsort()[:num_compare]
      best_acc = 0.
      best_loss = 100.
      accs = []
      losses = []
      for idx in idxes:
        pos_r = int(idx / C)
        pos_c = int(idx - pos_r*C)
        if not can_prune_pos(mask_reduce, pos_r, pos_c): continue
#        if mask_reduce[pos_r, pos_c] == 0: continue
#        print('debug:', pos_r, pos_c, s_reduce[idx], saliency[1].abs()[pos_r, pos_c])
        tmp_alpha = new_alpha_reduce[pos_r, pos_c].item()
        new_alpha_reduce[pos_r, pos_c].fill_(0.)
        criterion = nn.CrossEntropyLoss()
        acc, loss = infer(valid_queue, new_model, criterion, early_stop=args.iter_compare, verbose=False)
        del criterion
#        print('debug:', new_alpha_reduce)
        accs.append(acc)
        losses.append(loss)
#        if acc > best_acc:
        if loss < best_loss:
          best_pos_r = pos_r
          best_pos_c = pos_c
          best_acc = acc
          best_loss = loss
        new_alpha_reduce[pos_r, pos_c].fill_(tmp_alpha)
#        print('debug:', new_alpha_reduce)
      mask_reduce[best_pos_r, best_pos_c] = 0
      sal_change = s_reduce[best_pos_r*C+best_pos_c]
      print("Prune reduce idx: ", best_pos_r, best_pos_c)
      print('accs', accs)
      print('losses', losses)
      print("Saliency and beta and best acc & loss: ", s_reduce[best_pos_r*C+best_pos_c], new_alpha_reduce[best_pos_r, best_pos_c], best_acc, best_loss)
      del new_model
#      assert 0, "Need to debug"

  print("After pruning, saliency changed %f"%(sal_change))
  prune_masks = [mask_normal, mask_reduce]
  model.prune(prune_masks)
  model.update_softmax_arch_parameters()
  if optim_e > 0:
    # fine-tune
    print("Fine-tune begin")
    fine_tune(model, optim_e, train_queue, valid_queue)

  return prune_masks

def fine_tune(model, epochs, train_queue, valid_queue):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
        model.parameters(),
        0.01,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

  for e in range(epochs):
    for step, (input, target) in enumerate(train_queue):
      model.train()
      n = input.size(0)  #input.size :[32,3,32,32]
  
      input = Variable(input, requires_grad=False).cuda()
      target = Variable(target, requires_grad=False).cuda()
  
      optimizer.zero_grad()
      logits= model(input) 
  
      loss = criterion(logits, target)
  
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()
  
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)
  
      if step % args.report_freq == 0:
        logging.info('fine-tune %03d(%d/%d epoch) %e %f %f', step, e, epochs, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

if __name__ == '__main__':
  space = spaces_dict[args.space]
  main(space)

