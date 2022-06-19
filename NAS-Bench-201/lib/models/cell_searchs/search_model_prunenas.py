##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
########################################################
# DARTS: Differentiable Architecture Search, ICLR 2019 #
########################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from ..cell_operations import ResNetBasicblock
from .search_cells     import NAS201SearchCell as SearchCell
from .genotypes        import Structure


class TinyNetworkPruneNAS(nn.Module):

  def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
    super(TinyNetworkPruneNAS, self).__init__()
    self._C        = C
    self._layerN   = N
    self.max_nodes = max_nodes

    self._num_classes = num_classes
    self._search_space = search_space
    self._affine = affine
    self._track_running_stats = track_running_stats

    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index = C, None, None
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self.op_names   = deepcopy( search_space )
    self._Layer     = len(self.cells)
    self.edge2index = edge2index
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.arch_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(search_space)) )
    self.prune_masks = [torch.ones_like(self.arch_parameters).cuda().byte()] # 0-prune; 1-reserve

  def get_weights(self):
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

  def get_alphas(self):
    return [self.arch_parameters]

  def get_softmax_arch_parameters(self):
    return [self.softmax_arch_parameters]

  def update_softmax_arch_parameters(self, set_zero=False):
    if set_zero:
      mask = self.prune_masks[0]
  
      self.softmax_arch_parameters_tmp = torch.zeros_like(self.arch_parameters)
  
      for i in range(self.arch_parameters.shape[0]):
        if mask[i].sum() > 0:
          self.softmax_arch_parameters_tmp[i, mask[i]] = F.softmax(self.arch_parameters[i, normal_mask[i]], dim=-1)
  
      self.softmax_arch_parameters = self.softmax_arch_parameters_tmp

    else:
      self.softmax_arch_parameters = F.softmax(self.arch_parameters, dim=-1)

    return [self.softmax_arch_parameters]

  def show_alphas(self):
    with torch.no_grad():
#      return 'arch-parameters :\n{:}'.format( nn.functional.softmax(self.arch_parameters, dim=-1).cpu() )
      return 'arch-parameters :\n{:}'.format( self.softmax_arch_parameters.cpu() )

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def genotype(self):
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        with torch.no_grad():
          weights = self.softmax_arch_parameters[ self.edge2index[node_str] ] * self.prune_masks[0][ self.edge2index[node_str] ].float()
          op_name = self.op_names[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure( genotypes )

  def forward(self, inputs):
#    alphas  = nn.functional.softmax(self.arch_parameters, dim=-1)
    if self.softmax_arch_parameters is None:
      self.update_softmax_arch_parameters()
    alphas = self.softmax_arch_parameters

    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        feature = cell(feature, alphas)
      else:
        feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits

  def has_redundent_op(self, no_restrict=False):
    if no_restrict:
      if self.prune_masks[0].sum() > self.num_kept_min: return True
      return False

    for i in range(self.max_nodes):
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        if self.prune_masks[0][self.edge2index[node_str]].sum() > 1: return True
    return False

  def prune(self, prune_masks):
    self.prune_masks = prune_masks
    mask = prune_masks[0]

    for idx, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        for i in range(1, self.max_nodes):
          for j in range(i):
            node_str = '{:}<-{:}'.format(i, j)
            edge_mask = mask[self.edge2index[node_str]]
            for c, m in enumerate(edge_mask):
                if m == 0: cell.edges[node_str][c] = None
    torch.cuda.empty_cache()

  def new(self):
    model_new = TinyNetworkPruneNAS(self._C, self._layerN, self.max_nodes, self._num_classes, self._search_space, self._affine, self._track_running_stats).cuda()
    model_new.prune(self.prune_masks)
    for x, y in zip(model_new.get_alphas(), self.get_alphas()):
        x.data.copy_(y.data)
    return model_new
