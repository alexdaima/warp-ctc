from __future__ import division
import torch
from _ext import ctc
from .validators import validate_inputs


class CTCLoss(torch.autograd.Function):

    def __init__(self, take_average=True, blank=None):
        super(CTCLoss, self).__init__()
        self.take_average = take_average
        self.blank = blank

    def forward(self, activations, labels, lengths, label_lengths):
        use_cuda = activations.is_cuda
        validate_inputs(activations, labels, lengths, label_lengths)
        costs = torch.zeros(activations.size()[0])
        activations = torch.transpose(activations, 0, 1).contiguous()
        grads = activations.new(activations.size()).zero_()
        blank = self.blank
        if blank is None:
            blank = activations.size()[-1] - 1
        if use_cuda:
            ctc.ctc_cost_and_grad_cuda(activations, labels, lengths, label_lengths, costs, grads, blank)
        else:
            ctc.ctc_cost_and_grad(activations, labels, lengths, label_lengths, costs, grads, blank)
        self._grads = grads
        if use_cuda:
            costs = costs.cuda()
        cost = torch.sum(costs)
        if self.take_average:
            cost = cost / costs.size(0)
        return costs.new((cost,))

    def backward(self, *args):
        grads = self._grads.transpose_(0, 1).contiguous()
        if self.take_average:
            grads = grads / grads.size()[0]
        return grads, None, None, None
