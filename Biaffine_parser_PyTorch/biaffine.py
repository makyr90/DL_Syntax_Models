import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class Biaffine(nn.Module):
    """
    https://github.com/chantera/teras/blob/master/teras/framework/pytorch/model.py
    """

    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self._use_bias = bias

        shape = (in1_features + int(bias[0]),
                 in2_features + int(bias[1]),
                 out_features)
        self.weight = nn.Parameter(torch.Tensor(*shape))
        if bias[2]:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        self.weight.data.fill_(0.0)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, input1, input2):

        is_cuda = next(self.parameters()).is_cuda
        device_id = next(self.parameters()).get_device() if is_cuda else None
        out_size = self.out_features
        batch_size, len1, dim1 = input1.size()
        if self._use_bias[0]:
            ones = torch.ones(batch_size, len1, 1)
            if is_cuda:
                ones = ones.cuda(device_id)
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        len2, dim2 = input2.size()[1:]
        if self._use_bias[1]:
            ones = torch.ones(batch_size, len2, 1)
            if is_cuda:
                ones = ones.cuda(device_id)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1
        input1_reshaped = input1.contiguous().view(batch_size * len1, dim1)
        W_reshaped = torch.transpose(self.weight, 1, 2) \
            .contiguous().view(dim1, out_size * dim2)
        affine = torch.mm(input1_reshaped, W_reshaped) \
            .view(batch_size, len1 * out_size, dim2)
        biaffine = torch.transpose(
            torch.bmm(affine, torch.transpose(input2, 1, 2))
            .view(batch_size, len1, out_size, len2), 2, 3)
        if self._use_bias[2]:
            biaffine += self.bias.expand_as(biaffine)
        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'
