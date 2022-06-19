import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

import numpy as np

"""
cell 0 input: torch.Size([32, 48, 32, 32]) torch.Size([32, 48, 32, 32])
cell 0 output: torch.Size([32, 64, 32, 32])
cell 1 input: torch.Size([32, 48, 32, 32]) torch.Size([32, 64, 32, 32])
cell 1 output: torch.Size([32, 64, 32, 32])
cell 2 input: torch.Size([32, 64, 32, 32]) torch.Size([32, 64, 32, 32])
cell 2 output: torch.Size([32, 128, 16, 16])
cell 3 input: torch.Size([32, 64, 32, 32]) torch.Size([32, 128, 16, 16])
cell 3 output: torch.Size([32, 128, 16, 16])
cell 4 input: torch.Size([32, 128, 16, 16]) torch.Size([32, 128, 16, 16])
cell 4 output: torch.Size([32, 128, 16, 16])
cell 5 input: torch.Size([32, 128, 16, 16]) torch.Size([32, 128, 16, 16])
cell 5 output: torch.Size([32, 256, 8, 8])
cell 6 input: torch.Size([32, 128, 16, 16]) torch.Size([32, 256, 8, 8])
cell 6 output: torch.Size([32, 256, 8, 8])
cell 7 input: torch.Size([32, 256, 8, 8]) torch.Size([32, 256, 8, 8])
cell 7 output: torch.Size([32, 256, 8, 8])
"""

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
#    return sum(w * op(x) for w, op in zip(weights, self._ops))
    return sum(w * op(x) if op is not None else w*0 for w, op in zip(weights, self._ops))
#    out = []
#    for w, op in zip(weights, self._ops):
#      if op is not None:
#        out.append(w * op(x))
#    if len(out) > 0: return sum(out)
#    return None


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
#      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
#      s = []
#      for j, h in enumerate(states):
#        tmp = self._ops[offset+j](h, weights[offset+j])
#        if tmp is not None: s.append(tmp)
#      s = sum(s)
      offset += len(states)
#      if not torch.is_tensor(s):
      if len(s.shape) < 4:
        if self.reduction: 
          H,W = s0.shape[2:4]
          s = torch.zeros([s0.shape[0], s0.shape[1], int(H//2), int(W//2)], device=s0.device, dtype=s0.dtype) 
        else: 
          s = torch.zeros_like(s0)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, alpha_weights=None):
    super(Network, self).__init__()
    self.alpha_weights = alpha_weights
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()
#    self.prune_masks = [torch.ones_like(self.alphas_normal).byte(), torch.ones_like(self.alphas_reduce).byte()] # 0-prune; 1-reserve
    self.prune_masks = [torch.ones_like(self.alphas_normal).bool(), torch.ones_like(self.alphas_reduce).bool()] # 0-prune; 1-reserve

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    model_new.prune(self.prune_masks)
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input, student_output_idx=[]):
    """
    INPUT:
    	student_output_idx: the index of the cell, whose output will be add to outputs. '-1' means the output of stem; cell-2 and cell-5 are the reduced cells for the one-shot model stacked by 8 cells.

    OUTPUT:
        a list of feature maps, with len(student_output_idx)+1 items, the last item is the final output before the softmax.
    """
    outputs = []
    s0 = s1 = self.stem(input)

#    self.softmax_alphas_reduce = F.softmax(self.alphas_reduce, dim=-1)
#    self.softmax_alphas_normal = F.softmax(self.alphas_normal, dim=-1)
#    self.update_softmax_arch_parameters()

    if -1 in student_output_idx:
      outputs.append(s0)
    for i, cell in enumerate(self.cells):
      if self.alpha_weights is None:
        if cell.reduction:
          weights = self.softmax_alphas_reduce
        else:
          weights = self.softmax_alphas_normal
      else:
        raise(ValueError("Why you want to set alphas manually?"))
        print(self.alpha_weights['alphas_normal'])
        print(self.alpha_weights['alphas_reduce'])
        if cell.reduction:
          weights = self.alpha_weights['alphas_reduce']
        else:
          weights = self.alpha_weights['alphas_normal']
      s0, s1 = s1, cell(s0, s1, weights)
      if i in student_output_idx:
        outputs.append(s1)

    out = self.global_pooling(s1)  #[32,256,1,1]
    logits = self.classifier(out.view(out.size(0),-1))
    outputs.append(logits)
    return outputs

  def _loss(self, input, target):
    logits = self(input)[-1]
    return self._criterion(logits, target) 

  def save_grad(self, grad):
    print(grad)
    print('aaaaa')
    self.grad_softmax_alpha = grad
  def build_softmax_alpha_hook(self):
    self.grad_softmax_alpha = []
    self.hook_normal = self.softmax_alphas_normal.register_hook(self.save_grad)
  def remove_softmax_alpha_hook(self):
    self.hook_normal.remove()

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]
#    self.softmax_alphas_reduce = F.softmax(self.alphas_reduce, dim=-1)
#    self.softmax_alphas_normal = F.softmax(self.alphas_normal, dim=-1)

  def arch_parameters(self):
    return self._arch_parameters

  def get_softmax_arch_parameters(self):
    return [self.softmax_alphas_normal, self.softmax_alphas_reduce]

  def update_softmax_arch_parameters(self, set_zero=False):
    if set_zero:
      normal_mask, reduce_mask = self.prune_masks
  
      self.softmax_alphas_reduce_tmp = torch.zeros_like(self.alphas_reduce)
      self.softmax_alphas_normal_tmp = torch.zeros_like(self.alphas_normal)
#      softmax_alphas_normal = self.softmax_alphas_normal * normal_mask.float()
#      softmax_alphas_reduce = self.softmax_alphas_reduce * reduce_mask.float()
  
      for i in range(self.alphas_normal.shape[0]):
        if normal_mask[i].sum() > 0:
          self.softmax_alphas_normal_tmp[i, normal_mask[i]] = F.softmax(self.alphas_normal[i, normal_mask[i]], dim=-1)
        if reduce_mask[i].sum() > 0:
          self.softmax_alphas_reduce_tmp[i, reduce_mask[i]] = F.softmax(self.alphas_reduce[i, reduce_mask[i]], dim=-1)
  
      self.softmax_alphas_normal = self.softmax_alphas_normal_tmp
      self.softmax_alphas_reduce = self.softmax_alphas_reduce_tmp

    else:
      self.softmax_alphas_normal = F.softmax(self.alphas_normal, dim=-1)
      self.softmax_alphas_reduce = F.softmax(self.alphas_reduce, dim=-1)

    return [self.softmax_alphas_normal, self.softmax_alphas_reduce]

#
#    return [self.softmax_alphas_normal, self.softmax_alphas_reduce]

  def genotype(self, no_restrict=False):

    def _parse(weights, prune_mask, no_restrict=False):
      gene = []
      if no_restrict:
        activate = [1,1] + [0]*self._steps
        n = 2
        start = 0
        node_end_row = [] # [2,5,9,14]
        node_start_row = [] # [0,2,5,9]
        for i in range(self._steps): 
          end = start + n
          node_start_row.append(start)
          node_end_row.append(end)
          start = end
          n += 1
        R,C = weights.shape
        weights = (weights * prune_mask).reshape(-1)
        kept = int(min(prune_mask.sum().item(), self.num_kept_max))
        idxes = np.sort(weights.argsort()[-kept:])
        for idx in idxes:
          pos_r = int(idx / C)
          pos_c = int(idx - pos_r*C)
          if prune_mask[pos_r, pos_c] == 0: 
            print("idx %d-%d has been pruned, so we donot keep this op"%(pos_r, pos_c))
            continue
#            raise(ValueError("Code goes wrong"))
          for i, end_row in enumerate(node_end_row):
            if pos_r < end_row:
              dst_node = i+2
              break
          src_node = pos_r - node_start_row[dst_node-2]
          if not activate[src_node]: continue
          activate[dst_node] = 1
          gene.append((PRIMITIVES[pos_c], src_node, dst_node))
        concat = (np.where(activate[-self._multiplier:])[0] + 2).tolist()
            
      else:
        n = 2
        start = 0
        try:
          none_idx = PRIMITIVES.index('none')
        except:
          none_idx = -1
        for i in range(self._steps):
          end = start + n
          W = weights[start:end].copy()
          W = W * prune_mask[start:end]
          edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != none_idx))[:2]
          for j in edges:
            k_best = None
            for k in range(len(W[j])):
              if k != none_idx:
                if k_best is None or W[j][k] > W[j][k_best]:
                  k_best = k
            gene.append((PRIMITIVES[k_best], j))
          start = end
          n += 1
        concat = list(range(2+self._steps-self._multiplier, self._steps+2))
      return gene, concat

#    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
#    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
#    self.update_softmax_arch_parameters()
    gene_normal, concat_normal = _parse(self.softmax_alphas_normal.data.cpu().numpy(), self.prune_masks[0].data.cpu().numpy(), no_restrict)
    gene_reduce, concat_reduce = _parse(self.softmax_alphas_reduce.data.cpu().numpy(), self.prune_masks[1].data.cpu().numpy(), no_restrict)

    genotype = Genotype(
      normal=gene_normal, normal_concat=concat_normal,
      reduce=gene_reduce, reduce_concat=concat_reduce
    )
    return genotype

  def compute_sal_loss_for_hard_discretization(self, saliency):

    def _parse(weights, sal, prune_mask):
      gene = []
      n = 2
      start = 0
      try:
        none_idx = PRIMITIVES.index('none')
      except:
        none_idx = -1
      tmp_loss = sal.sum()
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != none_idx))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if prune_mask[j, k] == 0: continue
            if k != none_idx:
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          tmp_loss = tmp_loss - sal[j+start, k_best]
        start = end
        n += 1
      return tmp_loss

#    self.update_softmax_arch_parameters()
    normal_loss = _parse(self.softmax_alphas_normal.data.cpu().numpy(), saliency[0], self.prune_masks[0].data.cpu().numpy())
    reduce_loss = _parse(self.softmax_alphas_reduce.data.cpu().numpy(), saliency[1], self.prune_masks[1].data.cpu().numpy())
    print(normal_loss, reduce_loss)

    return normal_loss + reduce_loss

  def genotype_sal(self, saliencys):

    def _greedy(sal1, sal1_mask, sal2=None, sal2_mask=None, prev_loss=0, prev_coord=[], save_num=-1, degree=2):
      if not isinstance(prev_loss, list):
        prev_loss = [prev_loss]
      prev_loss = np.array(prev_loss)
      # construct all co-ordinations for one sal
      num_rows, num_ops = sal1.shape
      tmp_rows = list(itertools.product(range(num_rows), repeat=degree))
      all_rows = []
      for row_index in tmp_rows:
        # prune repeated and un-valid row-index
        if len(set(row_index)) < len(row_index): 
          continue
        prune = False
        for i, idx in enumerate(row_index[:-1]):
          if idx>row_index[i+1]: prune=True; break
        if prune: continue
        else: all_rows.append(row_index)
      assert(len(all_rows) == (math.factorial(num_rows)//(math.factorial(degree)*math.factorial(num_rows-degree))))
      all_cols = list(itertools.product(range(num_ops), repeat=degree))

      if sal2 is not None:
        # construct co-ordinations for two sals
        all_rows = list(itertools.product(all_rows, repeat=2))
        all_cols = list(itertools.product(all_cols, repeat=2))

      min_losses = []
      min_coords = []
      genes = []
      total_sal = sal1.sum() + sal2.sum() if sal2 is not None else sal1.sum()
      for row_idx in all_rows:
        for col_idx in all_cols:
          # judge wheter one of the chosen ops has been pruned
          if sal2 is not None:
              judge = (1-sal1_mask[row_idx[0], col_idx[0]]).sum() + (1-sal2_mask[row_idx[1], col_idx[1]]).sum()
          else:
              judge = (1-sal1_mask[row_idx, col_idx]).sum()
          if judge > 0: continue

          tmp_loss = total_sal - sal1[row_idx[0], col_idx[0]].sum() - sal2[row_idx[1], col_idx[1]].sum() if sal2 is not None else total_sal - sal1[row_idx, col_idx].sum()
          tmp_loss = prev_loss + tmp_loss
          min_idx = np.abs(tmp_loss).argmin()          
          min_loss = tmp_loss[min_idx]
#          min_loss = min(prev_loss, key=lambda x: abs(x+tmp_loss))
          min_losses.append(min_loss)
          if len(prev_coord) > 0:
            min_coords.append(prev_coord[min_idx]+[(row_idx, col_idx)])
          else: min_coords.append([(row_idx, col_idx)])
      assert(len(min_losses) == len(min_coords))
      num_losses = len(min_losses)
      if save_num < 0: save_num = num_losses
      if save_num >= num_losses:
        pass
#        tmp_idx = np.argpartition(np.abs(np.array(min_losses)), num_losses)
#        min_losses = [min_losses[idx] for idx in tmp_idx]
#        min_coords = [min_coords[idx] for idx in tmp_idx]
      else:
        tmp_idx = np.argpartition(np.abs(np.array(min_losses)), save_num)[:save_num]
        min_losses = [min_losses[idx] for idx in tmp_idx]
        min_coords = [min_coords[idx] for idx in tmp_idx]
      return min_losses, min_coords

    normal_sal, reduce_sal = saliencys
    normal_mask, reduce_mask = self.prune_masks

    gene_normal = []
    gene_reduce = []
    n = 2
    degree = 2
    start = 0
    min_losses = 0
    min_coords = []
    for i in range(self._steps):
      end = start + n
      print(i)
      min_losses, min_coords = _greedy(normal_sal[start:end].cpu().data.numpy(), normal_mask[start:end].cpu().data.numpy(),
                   reduce_sal[start:end].cpu().data.numpy(), reduce_mask[start:end].cpu().data.numpy(),
                   min_losses, min_coords, save_num=100, degree=degree)
      start = end
      n += 1

    # obtain genotype
    gene_normal = []
    gene_reduce = []
    min_idx = np.abs(min_losses).argmin()
    min_coord = min_coords[min_idx]
    for step, coord in enumerate(min_coord):
      row_idx, col_idx = coord
      for i in range(degree):
        gene_normal.append((PRIMITIVES[col_idx[0][i]], row_idx[0][i]))
        gene_reduce.append((PRIMITIVES[col_idx[1][i]], row_idx[1][i]))

    concat = list(range(2+self._steps-self._multiplier, self._steps+2))
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    geno_loss = min_losses[min_idx]
    return genotype, geno_loss

  def customized_weight(self, weights, numIn_per_node=2, choose_type='argmax'):
    """
    There are four types: argmax_operator, argmax_edge, sampe_operator, sample_edge
    In my opinion, the key should be argmax_edge & sample_edge

    argmax_operator: use argmax to choose the operator on each edge. But the input of each node is still full-sized.
    sample_operator: sample the operator on each edge. But the input of each node is still full-sized.

    argmax_edge: use argmax to choose the input for each node, default is 2, controlled by numIn_per_node
    sample_edge: sample the input for each node, default is 2, controlled by numIn_per_node
    """
    new_weights = torch.zeros_like(weights)
    weights_np = weights.data.cpu().numpy()
    if choose_type == 'argmax_edge':
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights_np[start:end].copy()
        actual_numIn = min(numIn_per_node, i+2)
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:actual_numIn]
        for j in edges:
          new_weights[start+j,:] = weights[start+j,:]
        start = end
        n += 1

    elif choose_type == 'sample_edge':
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights_np[start:end].copy()
        actual_numIn = min(numIn_per_node, i+2)
        p = np.max(W, axis=1)
        p = np.exp(p)
        p_sum = np.sum(p)
        p = p / p_sum
        edges = np.random.choice(range(i+2), actual_numIn, replace=False, p=p)
        for j in edges:
          new_weights[start+j,:] = weights[start+j,:]
        start = end
        n += 1

    elif choose_type == 'argmax_operator':
#      max_idx = np.argmax(weights_np, axis=1)
      max_idx = np.argpartition(-weights_np, 2, axis=1)[:,:2]
      for i in range(weights_np.shape[0]):  
        if max_idx[i][0] != PRIMITIVES.index('none'):
            new_weights[i, max_idx[i][0]] = weights[i, max_idx[i][0]]
        else:
            new_weights[i, max_idx[i][1]] = weights[i, max_idx[i][1]]

    elif choose_type == 'sample_operator':
      for i in range(weights_np.shape[0]):  
        idx = np.random.choice(range(weights_np.shape[1]), 2, replace=False, p=weights_np[i,:])
        if idx[0] != PRIMITIVES.index('none'):
            new_weights[i, idx[0]] = weights[i, idx[0]]
        else:
            new_weights[i, idx[1]] = weights[i, idx[1]]

    else:
      raise(ValueError('No type completed for type %s'%choose_type))
    return new_weights

  def customized_forward(self, input, numIn_per_node=2, choose_type='argmax_edge'):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)

      # get customized weights
      weights = self.customized_weight(weights, numIn_per_node=numIn_per_node, choose_type=choose_type)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def has_redundent_op(self, no_restrict=False):
    if no_restrict:
      if self.prune_masks[0].sum() > self.num_kept_min: return True
      if self.prune_masks[1].sum() > self.num_kept_min: return True
      return False

    n = 2
    start = 0
    for i in range(self._steps):
      end = start + n
      tmp = self.prune_masks[0][start:end]
      if tmp.sum() > 2: return True
      tmp = self.prune_masks[1][start:end]
      if tmp.sum() > 2: return True
      start = end
      n += 1
    return False

  def prune(self, prune_masks):
    self.prune_masks = prune_masks
    mask_normal, mask_reduce = prune_masks

    for idx, cell in enumerate(self.cells):
        mask = mask_reduce if cell.reduction else mask_normal
        r = 0
        for i in range(self._steps):
          for j in range(2+i):
            stride = 2 if cell.reduction and j < 2 else 1
            for c in range(mask.shape[1]):
              if mask[r,c] == 0:
                cell._ops[r]._ops[c] = None
            r = r+1
    torch.cuda.empty_cache()
