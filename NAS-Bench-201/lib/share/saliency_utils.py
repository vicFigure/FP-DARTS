import torch
from torch.autograd import Variable

def zero_grads(parameters):
    for p in parameters:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
            #if p.grad.volatile:
            #    p.grad.data.zero_()
            #else:
            #    data = p.grad.data
            #    p.grad = Variable(data.new().resize_as_(data).zero_()).

def gradient(_outputs, _inputs, grad_outputs=None, retain_graph=None,
            create_graph=False):
    if torch.is_tensor(_inputs):
        _inputs = [_inputs]
    else:
        _inputs = list(_inputs)
    grads = torch.autograd.grad(_outputs, _inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads,
                                                                         _inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])

def _hessian(outputs, inputs, prune_masks=None, weight_decay=0, out=None, allow_unused=False,
             create_graph=False):
    #assert outputs.data.ndimension() == 1

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = Variable(torch.zeros(n, n)).type_as(outputs)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(outputs, inp, create_graph=True,
                                     allow_unused=allow_unused)
        grad = grad.contiguous().view(-1) + weight_decay*inp.view(-1)
        #grad = outputs[i].contiguous().view(-1)
        if prune_masks is not None:
          mask = prune_masks[i].data.view(-1)
        else: mask = None
     
        for j in range(inp.numel()):
            if mask is not None and mask[j] == 0: 
                ai += 1
                continue
            # print('(i, j): ', i, j)
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], retain_graph=True)[j:]
            else:
                n = sum(x.numel() for x in inputs[i:]) - j
                row = Variable(torch.zeros(n)).type_as(grad[j])
                #row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out.data[ai, ai:].add_(row.clone().type_as(out).data)  # ai's row
            if ai + 1 < n:
                out.data[ai + 1:, ai].add_(row.clone().type_as(out).data[1:])  # ai's column
            del row
            ai += 1
        del grad
    return out
