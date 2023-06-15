from nfts.models.resnet import resnet18, tsa_resnet18
from nfts.models.vit import ett_vit_small, vit_small

_MODEL_LOOKUP = {
    "resnet18": resnet18,
    "tsa_resnet18": tsa_resnet18,
    "vit_small": vit_small,
    "ett_vit_small": ett_vit_small,
}


def model_factory(backbone, initialization, setting):
    return _MODEL_LOOKUP[backbone](initialization, setting)
