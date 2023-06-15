import os

import torch
import torch.nn as nn

from nfts.models.attack import adversarial_feature_perturbations
from nfts.models.protonet import ProtoNetHead
from nfts.models.supernet import ResNetSuperNetSampler
from nfts.models.tsa import FeatureAlignment, TSA_Conv2d


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        state_dict=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.transforms = None
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)
        if state_dict is not None:
            self.load_state_dict(state_dict, strict=True)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ProtoResNet(ResNet):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        state_dict=None,
    ):
        super().__init__(
            block,
            layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            state_dict=state_dict,
        )
        self.head = ProtoNetHead()

    def forward(self, support_data, support_targets, query_data, query_targets):
        support_embeddings = super().forward(support_data)
        query_embeddings = super().forward(query_data)
        return self.head(support_embeddings, support_targets, query_embeddings, query_targets)


def _resnet(state_dict, block, layers, **kwargs):
    model = ProtoResNet(block, layers, state_dict=state_dict, **kwargs)
    return model


def resnet18(initialization, setting, **kwargs):
    init_state_dict = torch.load(f"./models/{initialization}/resnet18.pth")
    model = _resnet(init_state_dict, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


class TSA_ResNet(ResNet):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        state_dict=None,
    ):
        super().__init__(
            block,
            layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            state_dict=state_dict,
        )
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                for name, m in block.named_children():
                    if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                        setattr(block, name, TSA_Conv2d(m))
        if hasattr(self.layer4[-1], "bn3"):
            num_features = self.layer4[-1].bn3.num_features
        else:
            num_features = self.layer4[-1].bn2.num_features
        self.fa = FeatureAlignment(num_features)

    def forward(self, x):
        x = super().forward(x)
        return self.fa(x)


class TSA_ProtoResNet(TSA_ResNet):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        state_dict=None,
    ):
        super().__init__(
            block,
            layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            state_dict=state_dict,
        )
        self.head = ProtoNetHead()

    def forward(self, support_data, support_targets, query_data, query_targets, afp=False):
        support_embeddings = super().forward(support_data)
        query_embeddings = super().forward(query_data)
        if afp:
            support_embeddings = adversarial_feature_perturbations(
                self.head, support_embeddings, support_targets, query_embeddings, query_targets
            )
        return self.head(support_embeddings, support_targets, query_embeddings, query_targets)


def _tsa_resnet(state_dict, block, layers, **kwargs):
    model = TSA_ProtoResNet(block, layers, state_dict=state_dict, **kwargs)
    num_decisions = len([n for n, m in model.named_modules() if isinstance(m, TSA_Conv2d)]) * 2
    model = ResNetSuperNetSampler(model, num_decisions)
    return model


def tsa_resnet18(initialization, setting, **kwargs):
    setting_state_dict_path = f"./models/nfts/tsa_resnet18_{setting}.pth"
    if os.path.exists(setting_state_dict_path) and os.path.isfile(setting_state_dict_path):
        setting_state_dict = torch.load(setting_state_dict_path, map_location="cpu")
        model = _tsa_resnet(None, BasicBlock, [2, 2, 2, 2], **kwargs)
        model.load_state_dict(setting_state_dict, strict=True)
    else:
        init_state_dict = torch.load(f"./models/{initialization}/resnet18.pth")
        model = _tsa_resnet(init_state_dict, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
