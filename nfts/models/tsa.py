import copy

import torch.nn as nn


class TSA_Conv2d(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op
        self.op_cp = copy.deepcopy(op)
        self.alpha = nn.Conv2d(
            self.op.in_channels,
            self.op.out_channels,
            kernel_size=self.op.kernel_size,
            stride=self.op.stride,
            padding=self.op.padding,
            bias=False,
        )
        nn.init.dirac_(self.alpha.weight)
        self.alpha.weight.data = self.alpha.weight.data * 0.0001
        self.do_adaptation = None
        self.do_finetuning = None

    def forward(self, x):
        assert self.do_adaptation is not None and self.do_finetuning is not None

        if self.do_finetuning:
            y = self.op_cp(x)
        else:
            y = self.op(x)

        if self.do_adaptation:
            return y + self.alpha(x)
        return y


class FeatureAlignment(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.beta = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False)
        nn.init.dirac_(self.beta.weight)

    def forward(self, x):
        if x.ndim == 2:
            x = x[:, :, None, None]
            x = self.beta(x)[:, :, 0, 0]
        else:
            x = self.beta(x)
        return x
