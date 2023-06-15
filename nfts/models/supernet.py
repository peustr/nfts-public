import numpy as np
import torch
import torch.nn as nn

from nfts.models.ett import Block_Ada
from nfts.models.tsa import FeatureAlignment, TSA_Conv2d
from nfts.utils import disable_grad, enable_grad


def sample_adapter_configuration(num_decisions):
    return np.random.rand(num_decisions).round().astype(np.uint8)


class ResNetSuperNetSampler(nn.Module):
    def __init__(self, model, num_decisions):
        super().__init__()
        self.model = model
        self.num_decisions = num_decisions

    def sample_path(self, adapter_configuration=None):
        if adapter_configuration is None:
            adapter_configuration = sample_adapter_configuration(self.num_decisions)
        disable_grad(self.model)
        i_conf = 0
        for _, m in self.model.named_modules():
            if isinstance(m, TSA_Conv2d):
                m.do_adaptation = bool(adapter_configuration[i_conf])
                m.do_finetuning = bool(adapter_configuration[i_conf + 1])
                if m.do_adaptation:
                    enable_grad(m.alpha)
                if m.do_finetuning:
                    enable_grad(m.op_cp)
                i_conf += 2
            elif isinstance(m, FeatureAlignment):
                enable_grad(m.beta)

    def forward(self, support_data, support_targets, query_data, query_targets):
        return self.model(support_data, support_targets, query_data, query_targets)


class ViTSuperNetSampler(nn.Module):
    def __init__(self, model, num_decisions):
        super().__init__()
        self.model = model
        self.num_decisions = num_decisions

    def sample_path(self, adapter_configuration=None):
        if adapter_configuration is None:
            adapter_configuration = sample_adapter_configuration(self.num_decisions)
        disable_grad(self.model)
        i_conf = 0
        for _, m in self.model.named_modules():
            if isinstance(m, Block_Ada):
                m.do_adaptation = bool(adapter_configuration[i_conf])
                m.do_finetuning = bool(adapter_configuration[i_conf + 1])
                if m.do_adaptation:
                    enable_grad(m.adapter_1)
                    enable_grad(m.adapter_2)
                if m.do_finetuning:
                    enable_grad(m.prefix_tuning)
                    enable_grad(m.feature_selection)
                    enable_grad(m.distillation_loss)
                i_conf += 2

    def forward(self, support_data, support_targets, query_data, query_targets, prefix=None):
        n_way = torch.unique(support_targets).shape[0]
        return self.model(support_data, support_targets, query_data, query_targets, n_way, prefix=prefix)
