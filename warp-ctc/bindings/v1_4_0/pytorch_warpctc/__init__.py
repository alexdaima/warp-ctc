from __future__ import division
import torch
import pytorch_warpctc
from ._warp_ctc import *
from .validators import validate_inputs


class CTCAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, activations, labels, lengths, label_lengths, take_average=True, blank=None):
        use_cuda = activations.is_cuda
        validate_inputs(activations, labels, lengths, label_lengths)
        costs = torch.zeros(activations.size()[0])
        activations = torch.transpose(activations, 0, 1).contiguous()
        grads = activations.new(activations.size()).zero_()
        batch_size = activations.shape[1]
        if blank is None:
            blank = activations.size()[-1] - 1
        if use_cuda:
            pytorch_warpctc.gpu_ctc(activations, grads, labels, label_lengths, lengths, batch_size, costs, blank)
        else:
            pytorch_warpctc.cpu_ctc(activations, grads, labels, label_lengths, lengths, batch_size, costs, blank)
        if use_cuda:
            costs = costs.cuda()
        cost = torch.sum(costs)
        grads = grads.transpose_(0, 1).contiguous()
        if take_average is True:
            cost = cost / costs.size(0)
            grads = grads / grads.size()[0]
        ctx.grads = grads
        return costs.new((cost,))

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grads, None, None, None, None, None


class CTCLoss(torch.nn.Module):

    def __init__(self, take_average=True, blank=None):
        super(CTCLoss, self).__init__()
        self.take_average = take_average
        self.blank = blank

    def forward(self, activations, labels, lengths, label_lengths):
        return CTCAutogradFunction.apply(
            activations,
            labels,
            lengths,
            label_lengths,
            self.take_average,
            self.blank
        )
