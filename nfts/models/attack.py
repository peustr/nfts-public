import copy

import torch
import torch.nn.functional as F


def fgsm(model, support_data, support_targets, query_data, query_targets, epsilon):
    return pgd(
        model,
        support_data,
        support_targets,
        query_data,
        query_targets,
        epsilon=epsilon,
        num_iter=1,
        step_size=1.0,
        random_start=False,
    )


def pgd(
    model,
    support_data,
    support_targets,
    query_data,
    query_targets,
    epsilon,
    num_iter=10,
    step_size=0.01,
    random_start=True,
):
    model.eval()
    perturbed_data = copy.deepcopy(support_data)
    perturbed_data.requires_grad = True
    data_min = (support_data - epsilon).clamp(-1.0, 1.0)
    data_max = (support_data + epsilon).clamp(-1.0, 1.0)
    if random_start:
        with torch.no_grad():
            perturbed_data.data += torch.empty_like(perturbed_data.data).uniform_(-epsilon, epsilon)
            perturbed_data.data.clamp_(data_min, data_max)
    for _ in range(num_iter):
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()
        loss = F.cross_entropy(model(perturbed_data, support_targets, query_data, query_targets), query_targets)
        loss.backward()
        with torch.no_grad():
            perturbed_data.data += step_size * perturbed_data.grad.data.sign()
            perturbed_data.data.clamp_(data_min, data_max)
    perturbed_data.requires_grad = False
    return perturbed_data


def adversarial_feature_perturbations(
    model,
    support_embeddings,
    support_targets,
    query_embeddings,
    query_targets,
    step_size=0.01,
):
    loss = F.cross_entropy(model, support_embeddings, support_targets, query_embeddings, query_targets)
    loss.backward(retain_graph=True)
    support_embeddings = support_embeddings + step_size * support_embeddings.grad.data.sign()
    return support_embeddings
