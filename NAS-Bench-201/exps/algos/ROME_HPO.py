##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
###########################################################################
# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019 #
###########################################################################
import os
import sys, time, random, argparse
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces
from nas_201_api  import NASBench201API as API

import share

def sample_update_hp(hyperopt, optimizer):
  """
  sample and update hyper-parameters
  """
  hpo_sample = hyperopt.sample_hp()
#  lr = lr_list[hpo_sample[1][0]]
  lr = hpo_sample[0][0]
  beta1 = hpo_sample[0][1]
  beta2 = hpo_sample[0][2]
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  if hyperopt._hp_probs[1].trainable or hyperopt._hp_probs[2].trainable:
    optimizer.change_beta(beta1, beta2)
  return hpo_sample

def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, hyperopt, epoch_str, print_freq, logger, iter_w=8, iter_a=8):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.train()
  end = time.time()

  lr_list = scheduler.get_lr()
  hyperopt.change_means(lr_list, idx=0)
  hpo_sample = sample_update_hp(hyperopt, w_optimizer)
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
#    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    w_optimizer.zero_grad()
    hyperopt.init_gradient()
    for t in range(iter_w):
      # For HPO
      hpo_sample = sample_update_hp(hyperopt, w_optimizer)
      _, logits = network(base_inputs)
      base_loss = criterion(logits, base_targets)
      base_loss.backward()
      hyperopt.update_gradient(base_loss)
    nn.utils.clip_grad_norm_(network.parameters(), 5)
    w_optimizer.step()
    hyperopt.step()

    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(),  base_inputs.size(0))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))

    # update the architecture-weight
    a_optimizer.zero_grad()
    for t in range(iter_a):
      _, logits = network(arch_inputs)
      arch_loss = criterion(logits, arch_targets) / iter_a
      arch_loss.backward()
    a_optimizer.step()
    # record
    arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
    arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
    arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
    arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
      logger.log('Sampled hyper-parameters')
      logger.log('lr: {:.3f}'.format(hpo_sample[0][0]))
      logger.log('beta1: {:.3f}'.format(hpo_sample[0][1]))
      logger.log('beta2: {:.3f}'.format(hpo_sample[0][2]))
      logger.log('wd: {:.6f}'.format(hpo_sample[0][3]))
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  #config_path = 'configs/nas-benchmark/algos/ROME_HPO.config'
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', config.batch_size, xargs.workers)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces('cell', xargs.search_space_name)
  if xargs.model_config is None:
    model_config = dict2config({'name': 'ROME_HPO', 'C': xargs.channel, 'N': xargs.num_cells,
                                'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                'space'    : search_space,
                                'affine'   : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  else:
    model_config = load_config(xargs.model_config, {'num_classes': class_num, 'space'    : search_space,
                                                    'affine'     : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  search_model = get_cell_based_tiny_net(model_config)
  logger.log('search-model :\n{:}'.format(search_model))
  logger.log('model-config : {:}'.format(model_config))

  _, _, criterion = get_optim_scheduler(search_model.get_weights(), config)
  # HPO optimizer
#  lr_list = np.array([0.025])
#  lr_prob = share.PROB_CONFIG('delta', tau=1., trainable=False, config={})
  beta1_list = np.array([0.47])
  beta1_prob = share.PROB_CONFIG('delta', tau=1., trainable=False, config={})
  beta2_list = np.array([1.])
  beta2_prob = share.PROB_CONFIG('delta', tau=1., trainable=False, config={})
#  wd_list = np.array([3e-4])
#  wd_prob = share.PROB_CONFIG('delta', tau=1., trainable=False, config={})

  lr_list = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
  lr_prob = share.PROB_CONFIG('Gaussian', tau=1., trainable=True, config={'std':0.005})
#  beta1_list = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
#  beta1_prob = share.PROB_CONFIG('delta', tau=1., trainable=True, config={})
#  beta2_list = np.array([0.9, 0.99, 0.999, 0.9999, 1.])
#  beta2_prob = share.PROB_CONFIG('delta', tau=1., trainable=True, config={})
  wd_list = np.array([3e-2, 3e-3, 3e-4, 3e-5])
  wd_prob = share.PROB_CONFIG('Gaussian', tau=1., trainable=True, config={'std':[1e-5, 1e-5, 1e-5, 1e-5]})

  hyperopt = share.HPO(lr_list, lr_prob,
                 beta1_list, beta1_prob,
                 beta2_list, beta2_prob,
                 wd_list, wd_prob
             )
  w_optimizer = share.General_Adam(
      search_model.parameters(),
      config.LR,
      betas=(0.47, 1.),
      weight_decay=config.decay,
      normal_beta=True
      )
  T_max = getattr(config, 'T_max', config.epochs)
  w_scheduler = share.DecayScheduler(base_lr=lr_list, eta_min=config.eta_min, T_max=T_max, decay_type='cosineAnnealing')
  
  a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  flop, param  = get_model_infos(search_model, xshape)
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  logger.log('search-space [{:} ops] : {:}'.format(len(search_space), search_space))
  if xargs.arch_nas_dataset is None:
    api = None
  else:
    api = API(xargs.arch_nas_dataset)
  logger.log('{:} create API = {:} done'.format(time_string(), api))

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  model_base_path = os.path.join(xargs.save_dir, 'search_final.pth')
  network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()

  if last_info.exists(): # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info   = torch.load(last_info)
    start_epoch = last_info['epoch']
    checkpoint  = torch.load(last_info['last_checkpoint'])
    genotypes   = checkpoint['genotypes']
    valid_accuracies = checkpoint['valid_accuracies']
    search_model.load_state_dict( checkpoint['search_model'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {-1: search_model.genotype()}

  # start training
  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
  for epoch in range(start_epoch, total_epoch):
#    w_scheduler.update(epoch, 0.0)
    w_scheduler.step(epoch)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    search_model.set_tau( xargs.tau_max - (xargs.tau_max-xargs.tau_min) * epoch / (total_epoch-1) )
    logger.log('\n[Search the {:}-th epoch] {:}, tau={:}'.format(epoch_str, need_time, search_model.get_tau()))
    logger.log('Candidate lr')
    logger.log(lr_list)

    search_w_loss, search_w_top1, search_w_top5, valid_a_loss , valid_a_top1 , valid_a_top5 \
              = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, hyperopt, epoch_str, xargs.print_freq, logger, xargs.iter_w, xargs.iter_a)
    search_time.update(time.time() - start_time)
    logger.log('[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))
    logger.log('[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss , valid_a_top1 , valid_a_top5 ))
    # check the best accuracy
    valid_accuracies[epoch] = valid_a_top1
    if valid_a_top1 > valid_accuracies['best']:
      valid_accuracies['best'] = valid_a_top1
      genotypes['best']        = search_model.genotype()
      find_best = True
    else: find_best = False

    genotypes[epoch] = search_model.genotype()
    logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
    # save checkpoint
    save_path = save_checkpoint({'epoch' : epoch + 1,
                'args'  : deepcopy(xargs),
                'search_model': search_model.state_dict(),
                'w_optimizer' : w_optimizer.state_dict(),
                'a_optimizer' : a_optimizer.state_dict(),
                'hyper_params': hyperopt.best_hp(),
                'genotypes'   : genotypes,
                'valid_accuracies' : valid_accuracies},
                model_base_path, logger)
    last_info = save_checkpoint({
          'epoch': epoch + 1,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, logger.path('info'), logger)
    if find_best:
      logger.log('<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.'.format(epoch_str, valid_a_top1))
      logger.log(genotypes[epoch])
      logger.log(hyperopt.best_hp())
      copy_checkpoint(model_base_path, model_best_path, logger)
    with torch.no_grad():
      logger.log('{:}'.format(search_model.show_alphas()))
      # logging for HPO
      logger.log("-"*20)
      logger.log("lr_alphas")
      logger.log(hyperopt.lr_alpha)
      logger.log("Normalized HP alphas: lr; beta1; beta2, wd")
      logger.log(F.softmax(hyperopt.lr_alpha, dim=-1))
      logger.log(F.softmax(hyperopt.beta1_alpha, dim=-1))
      logger.log(F.softmax(hyperopt.beta2_alpha, dim=-1))
      logger.log(F.softmax(hyperopt.wd_alpha, dim=-1))
      logger.log("-"*20)
      logger.log("Best HP values: lr, beta1, beta2, wd")
      logger.log(hyperopt.best_hp())
      logger.log("="*20)

    if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[epoch], '200')))
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  logger.log('\n' + '-'*100)
  # check the performance from the architecture dataset
  logger.log('ROME_HPO : run {:} epochs, cost {:.1f} s, last-geno is {:}.'.format(total_epoch, search_time.sum, genotypes[total_epoch-1]))
  logger.log('last-hyper is {:}.'.format(hyperopt.best_hp()))
  if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[total_epoch-1], '200')))
  logger.close()
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser("ROME_HPO")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The path of the configuration.')
  parser.add_argument('--model_config',       type=str,   help='The path of the model configuration. When this arg is set, it will cover max_nodes / channels / num_cells.')
  # architecture leraning rate
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
  parser.add_argument('--tau_min',            type=float,               help='The minimum tau for Gumbel')
  parser.add_argument('--tau_max',            type=float,               help='The maximum tau for Gumbel')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--task', type=int, default=0, help='task ID')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  # for ROME
  parser.add_argument('--iter_w',            type=int, default=8,   help='The number of samples for each update of weight.')
  parser.add_argument('--iter_a',            type=int, default=8,   help='The number of samples for each update of alpha.')
  parser.add_argument('--gpu', type=int, default=0, help='gpu')
  args = parser.parse_args()

  args.save_dir = '{}/{}-task{}'.format(args.save_dir, time.strftime("%Y%m%d-%H%M%S"), args.task)
  print('Experiment dir', args.save_dir)
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
#  torch.cuda.set_device(args.gpu)
  main(args)
