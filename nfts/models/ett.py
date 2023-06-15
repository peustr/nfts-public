import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(
        self,
        in_dim=384,
        out_dim=64,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.projector = nn.Linear(in_dim, out_dim)
        self.init_prefix = nn.Parameter(torch.randn(in_dim))

    def forward(self, prefix):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        prefix = F.normalize(prefix, dim=-1, p=2)
        init_prefix = F.normalize(self.init_prefix, dim=-1, p=2)

        prefix = self.projector(prefix)
        init_prefix = self.projector(init_prefix)

        prefix = prefix / self.student_temp
        init_out = F.softmax(init_prefix - self.center, dim=-1).detach()

        loss = torch.sum(-init_out * F.log_softmax(prefix, dim=-1), dim=-1).mean()

        self.update_center(init_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class FeatureSelection(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.theta = nn.Linear(feature_dim, feature_dim)
        nn.init.eye_(self.theta.weight)

    def forward(self, x):
        return self.theta(x)


class PrefixTuning(nn.Module):
    def __init__(self, feature_dim, num_heads=6):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.hidden_dim = feature_dim // 2
        self.control_trans = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim), nn.Tanh(), nn.Linear(self.hidden_dim, 2 * feature_dim)
        )

    def forward(self, prefix, n_way):
        prefix = self.control_trans(prefix).view(n_way, 2, self.num_heads, self.feature_dim // self.num_heads)
        prefix = prefix.permute(1, 2, 0, 3)
        return prefix


class DRA(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        return self.offset


class Block_Ada(nn.Module):
    def __init__(self, op, dim=384, num_heads=6):
        super().__init__()
        self.op = op
        self.adapter_1 = DRA(dim)
        self.adapter_2 = DRA(dim)
        self.prefix_tuning = PrefixTuning(dim, num_heads=num_heads)
        self.feature_selection = FeatureSelection(dim)
        self.distillation_loss = DistillationLoss(dim)
        self.do_adaptation = None
        self.do_finetuning = None

    def forward(self, x, n_way, prefix=None, return_attention=False):
        assert prefix is not None
        assert self.do_adaptation is not None and self.do_finetuning is not None

        prefix_weight = prefix.clone().to(self.distillation_loss.projector.weight.device).requires_grad_(True)

        if self.do_finetuning and prefix is not None:
            prefix = self.prefix_tuning(prefix, n_way)
            y, attn = self.op.attn(self.op.norm1(x), prefix=prefix)
            y = self.feature_selection(y)
        else:
            y, attn = self.op.attn(self.op.norm1(x))

        if return_attention:
            return attn

        tmp = self.op.drop_path(y)
        if self.do_adaptation:
            tmp = tmp + self.adapter_1(tmp)
        x = x + tmp

        tmp = self.op.drop_path(self.op.mlp(self.op.norm2(x)))
        if self.do_adaptation:
            tmp = tmp + self.adapter_2(tmp)
        x = x + tmp

        if self.do_finetuning and prefix is not None:
            loss = self.distillation_loss(prefix_weight)
        else:
            loss = None

        return x, loss
