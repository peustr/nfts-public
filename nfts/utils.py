def enable_grad(module):
    for p in module.parameters():
        p.requires_grad = True


def disable_grad(module):
    for p in module.parameters():
        p.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
