##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
########################################################
# DARTS: Differentiable Architecture Search, ICLR 2019 #
########################################################
import os, sys, time, random, argparse
import json
from copy import deepcopy
import torch
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces
from nas_201_api  import NASBench201API as API
from share import saliency_utils


def search_func(xloader, sal_queue, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger, gradient_clip, prune_config, prune=False):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.train()
  end = time.time()
  interval_prune = int(len(xloader) / 2.)
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))

    base_inputs = base_inputs.cuda(non_blocking=True)
    arch_inputs = arch_inputs.cuda(non_blocking=True)
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)

    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    w_optimizer.zero_grad()
    _, logits = network(base_inputs)
    base_loss = criterion(logits, base_targets)
    base_loss.backward()
    if gradient_clip > 0: torch.nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
    w_optimizer.step()
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(),  base_inputs.size(0))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))

    network.update_softmax_arch_parameters()
    # update the architecture-weight
    a_optimizer.zero_grad()
    _, logits = network(arch_inputs)
    arch_loss = criterion(logits, arch_targets)
    arch_loss.backward()
    a_optimizer.step()
    network.update_softmax_arch_parameters()
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

    # compute saliency & prune
    if prune and (step+1)%interval_prune==0:
      optim_e = 0
      sal_file = os.path.join(prune_config['sal_dir'], '%s.txt'%epoch_str)
      torch.cuda.empty_cache()
      saliency = compute_sal(sal_queue, network, criterion, sal_type=prune_config['sal_type'], sal_file=sal_file, second_order=prune_config['sal_second'])
      torch.cuda.empty_cache()

      prune_masks = prune_by_mask(network, saliency, prune_config['num_compare'], optim_e, xloader, sal_queue, no_restrict=False, prune_config=prune_config)
      network.update_softmax_arch_parameters()
      print("-"*10)
      print("prune_masks:")
      print(prune_masks[0])
      print("-"*10)


  return base_losses.avg, base_top1.avg, base_top5.avg


def valid_func(xloader, network, criterion, early_stop=-1):
  data_time, batch_time = AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.eval()
  end = time.time()
  with torch.no_grad():
    for step, (arch_inputs, arch_targets) in enumerate(xloader):
      arch_inputs = arch_inputs.cuda(non_blocking=True)
      arch_targets = arch_targets.cuda(non_blocking=True)
      # measure data loading time
      data_time.update(time.time() - end)
      # prediction
      _, logits = network(arch_inputs)
      arch_loss = criterion(logits, arch_targets)
      # record
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
      arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
      arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))
      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()
      if early_stop > 0 and step >= early_stop: break
  return arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
#  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.enabled   = False
  torch.backends.cudnn.benchmark = False
#  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', config.batch_size, xargs.workers)
  sal_queue = deepcopy(valid_loader)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces('cell', xargs.search_space_name)
  if xargs.model_config is None:
    model_config = dict2config({'name': 'PruneNAS', 'C': xargs.channel, 'N': xargs.num_cells,
                                'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                'space'    : search_space,
                                'affine'   : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  else:
    model_config = load_config(xargs.model_config, {'num_classes': class_num, 'space'    : search_space,
                                                    'affine'     : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  search_model = get_cell_based_tiny_net(model_config)
  logger.log('search-model :\n{:}'.format(search_model))
  
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
  a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  search_model.update_softmax_arch_parameters()
  flop, param  = get_model_infos(search_model, xshape)
  #logger.log('{:}'.format(search_model))
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  if xargs.arch_nas_dataset is None:
    api = None
  else:
    api = API(xargs.arch_nas_dataset)
  logger.log('{:} create API = {:} done'.format(time_string(), api))

  last_info, model_base_path, model_best_path = logger.path('info'), str(logger.path('model')), str(logger.path('best'))
#  network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
  network, riterion = search_model.cuda(), criterion.cuda()
  network.update_softmax_arch_parameters()

  if last_info.exists(): # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info   = torch.load(str(last_info))
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
    start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {-1: network.genotype()}

  # start training
  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
  prune_config = {
           'sal_dir': os.path.join(str(xargs.save_dir), 'sal'),
           'sal_type': xargs.sal_type,
           'sal_second': xargs.sal_second,
           'num_compare': xargs.num_compare,
           'iter_compare': xargs.iter_compare,
           'reg_type': xargs.reg_type,
           'reg_alpha': xargs.reg_alpha,
          }
  print(prune_config)
  if not os.path.exists(prune_config['sal_dir']):
    os.makedirs(prune_config['sal_dir'])
  for epoch in range(start_epoch, total_epoch):
    w_scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr())))

    prune = epoch >= config.warmup and search_model.has_redundent_op()
    network.update_softmax_arch_parameters()
    search_w_loss, search_w_top1, search_w_top5 = search_func(search_loader, sal_queue, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs.print_freq, logger, xargs.gradient_clip, prune_config, prune=prune)
    search_time.update(time.time() - start_time)
    logger.log('[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))
    valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion)
    logger.log('[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5))
    # check the best accuracy
    valid_accuracies[epoch] = valid_a_top1
    if valid_a_top1 > valid_accuracies['best']:
      valid_accuracies['best'] = valid_a_top1
      genotypes['best']        = network.genotype()
      find_best = True
    else: find_best = False

    genotypes[epoch] = network.genotype()
    logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
    # save checkpoint
    save_path = save_checkpoint({'epoch' : epoch + 1,
                'args'  : deepcopy(xargs),
                'search_model': search_model.state_dict(),
                'w_optimizer' : w_optimizer.state_dict(),
                'a_optimizer' : a_optimizer.state_dict(),
                'w_scheduler' : w_scheduler.state_dict(),
                'genotypes'   : genotypes,
                'valid_accuracies' : valid_accuracies},
                str(model_base_path), logger)
    last_info = save_checkpoint({
          'epoch': epoch + 1,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, str(logger.path('info')), logger)
    if find_best:
      logger.log('<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.'.format(epoch_str, valid_a_top1))
      copy_checkpoint(str(model_base_path), str(model_best_path), logger)
    with torch.no_grad():
      #logger.log('arch-parameters :\n{:}'.format(torch.nn.functional.softmax(search_model.arch_parameters, dim=-1).cpu() ))
      logger.log('{:}'.format(search_model.show_alphas()))
    if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[epoch], '200')))
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  logger.log('\n' + '-'*100)
  logger.log('PruneNAS : run {:} epochs, cost {:.1f} s, last-geno is {:}.'.format(total_epoch, search_time.sum, genotypes[total_epoch-1]))
  if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[total_epoch-1], '200')))
  logger.close()
  

def compute_sal(valid_queue, model, criterion, sal_type='task', sal_file=None, second_order=False):
  ###################
  # vanilla saliency
  ###################
  if sal_type == 'vanilla':
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

  t_saliencys = [0]
  l_saliencys = [0]
  for step, (input, target) in enumerate(valid_queue):
#      input = Variable(input, volatile=True).cuda()
#      target = Variable(target, volatile=True).cuda()
      input = input.cuda()
      target = target.cuda()


      outputs = model(input)
      logits = outputs[-1]

      alphas = model.get_softmax_arch_parameters()
#      alphas = model.get_alphas()

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
        

      if step % 100 == 0:
        print('saliency step %03d/%d' % (step, early_stop))
      if early_stop > 0 and step >= early_stop: break


  saliency = [v/step for v in t_saliencys]
  print("Saliency")
  print(saliency)
  # save saliency
  if sal_file is not None:
    save_sal = {}
    save_sal['normal'] = saliency[0].data.cpu().numpy().tolist()
    with open(sal_file, 'w') as f:
      json.dump(save_sal, f)
  return saliency
   

def prune_by_mask(model, saliency, num_compare=1, optim_e=0, train_queue=None, valid_queue=None, no_restrict=False, prune_config=None):
  assert prune_config is not None

  def can_prune_pos(mask, pos_r, pos_c):
    if mask[pos_r, pos_c] == 0: return False
    if no_restrict: return True
    if mask[pos_r].sum() == 1: return False
    return True

  def refine_sal(mask, sal, max_value):
    ''' Since DARTS add some restriction on the degree of each node, so if a node has and only has two edges, we cannot prune the edges to this node anymore'''
    col_sum = mask.sum(dim=1)
    num_valid_row = (col_sum>0).sum()
    for r in range(mask.shape[0]):
      if col_sum[r] == 1: sal[r].fill_(max_value)
    return sal

  R,C = saliency[0].shape
  mask_normal = model.prune_masks[0]
  s_normal = saliency[0].abs()
  max_normal = s_normal.max()
  # do not prune the same edge repeatedly
  s_normal = torch.where(mask_normal==1, s_normal, max_normal)
  if not no_restrict:
    # make sure each super-edge owns at least 1 operation
    s_normal = refine_sal(mask_normal, s_normal, max_normal)

  alpha_normal = model.get_softmax_arch_parameters()[0]
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

  else:
      new_model = model.new()
      model_dict = model.state_dict()
      new_model.load_state_dict(model_dict, strict=True)

      new_model.update_softmax_arch_parameters()
      new_alpha_normal = new_model.get_softmax_arch_parameters()[0]
      new_alpha_normal.data.mul_(mask_normal.float())

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
        criterion = torch.nn.CrossEntropyLoss()
        loss, acc, _ = valid_func(valid_queue, new_model, criterion, early_stop=prune_config['iter_compare'])
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
      del new_model
#      assert 0, "Need to debug"

  print("After pruning, saliency changed %f"%(sal_change))
  prune_masks = [mask_normal]
  model.prune(prune_masks)
  model.update_softmax_arch_parameters()
  if optim_e > 0:
    raise(ValueError("Not Implemented"))
    # fine-tune
    print("Fine-tune begin")

  return prune_masks

if __name__ == '__main__':
  parser = argparse.ArgumentParser("PruneNAS")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The config path.')
  parser.add_argument('--model_config',       type=str,   help='The path of the model configuration. When this arg is set, it will cover max_nodes / channels / num_cells.')
  parser.add_argument('--gradient_clip',      type=float, default=5, help='')
  # architecture leraning rate
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--task', type=int, default=0, help='task ID')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')

# For Prune
  parser.add_argument('--sal_type', type=str, default='task', help='type of saliency: task or vanilla')
  parser.add_argument('--sal_second', action='store_true', default=False, help='keep second order of Taylor Expansion')
  parser.add_argument('--num_compare', type=int, default=1, help='The number of candidate ops before pruning (The top least important in saliency). We need to calculate the loss and acc of these candidates and prune the least important one')
  parser.add_argument('--iter_compare', type=int, default=30, help='The iteration (batch) to run each comparison candidate')
  parser.add_argument('--reg_type', type=str, default='norm', help='type of regularization: norm, gini, or entropy')
  parser.add_argument('--reg_alpha', type=float, default=0.1, help='weight of KD_logits_Loss')

  args = parser.parse_args()
  args.save_dir = '{}/{}-task{}'.format(args.save_dir, time.strftime("%Y%m%d-%H%M%S"), args.task)
  print('Experiment dir', args.save_dir)
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
