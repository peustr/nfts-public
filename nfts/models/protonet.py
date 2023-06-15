import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNetHead(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _compute_prototypes(embeddings, targets, n_way):
        prototypes = torch.zeros(n_way, embeddings.shape[-1]).type(embeddings.dtype).to(embeddings.device)
        for i_target in range(n_way):
            prototypes[i_target] = embeddings[(targets == i_target).nonzero(), :].mean(0)
        return prototypes

    @staticmethod
    def _compute_distances(query_embeddings, prototypes, distance="cos"):
        assert distance in ("cos", "l2")
        if distance == "cos":
            logits = F.cosine_similarity(query_embeddings[:, None, :], prototypes, dim=-1, eps=1e-8) * 10
        else:
            logits = -torch.pow(query_embeddings[:, None, :] - prototypes, 2).sum(-1)
        return logits

    def forward(self, support_embeddings, support_targets, query_embeddings, query_targets):
        n_way = query_targets.unique().shape[0]
        prototypes = self._compute_prototypes(support_embeddings, support_targets, n_way)
        return self._compute_distances(query_embeddings, prototypes)
