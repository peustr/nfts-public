import copy
from collections import deque
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adadelta, Adam, AdamW

_OPTIMIZER_LOOKUP = {
    "adadelta": Adadelta,
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
}


def train(supernet, data_loader, args, path=None):
    if args.num_epochs != 1:
        raise ValueError("Number of epochs should be 1 during training.")
    if args.num_episodes >= 1000:
        window_len = 1000
    else:
        window_len = int(np.clip(args.num_episodes // 10, 1, 100))
    _optimizer = _init_optimizer(supernet, args.optimizer, {"lr": args.lr, "weight_decay": args.weight_decay})
    train_loss_avg, train_acc_avg = deque(maxlen=window_len), deque(maxlen=window_len)
    i_episode = 1
    for support_data, support_targets, query_data, query_targets in data_loader:
        support_data, support_targets, query_data, query_targets = _prepare_data(
            support_data, support_targets, query_data, query_targets, args.device
        )
        start_time = timer()
        if "vit" in args.backbone:
            prefix = _get_init_prefix(supernet, support_data, support_targets)
        else:
            prefix = None
        for j_epoch in range(args.num_epochs):
            train_loss, train_acc = _train_one_episode(
                supernet,
                _optimizer,
                support_data,
                support_targets,
                query_data,
                query_targets,
                args,
                path=path,
                prefix=prefix,
            )
        train_loss_avg.append(train_loss)
        train_acc_avg.append(train_acc)
        end_time = timer()
        interval = (end_time - start_time) / 60.0
        print(_episode_summary(i_episode, window_len, np.mean(train_loss_avg), np.mean(train_acc_avg), interval))
        i_episode += 1
        if i_episode >= args.num_episodes:
            break
    return np.mean(train_loss_avg), np.mean(train_acc_avg)


def validate(supernet, data_loader, args, path=None):
    model_ckpt = copy.deepcopy(supernet.state_dict())
    val_loss_avg, val_acc_avg = [], []
    i_episode = 1
    for support_data, support_targets, query_data, query_targets in data_loader:
        support_data, support_targets, query_data, query_targets = _prepare_data(
            support_data, support_targets, query_data, query_targets, args.device
        )
        _optimizer = _init_optimizer(supernet, args.optimizer, {"lr": args.lr, "weight_decay": args.weight_decay})
        if "vit" in args.backbone:
            prefix = _get_init_prefix(supernet, support_data, support_targets)
        else:
            prefix = None
        for j_epoch in range(args.num_epochs):
            train_loss, train_acc = _train_one_episode(
                supernet,
                _optimizer,
                support_data,
                support_targets,
                support_data,
                support_targets,
                args,
                path=path,
                prefix=prefix,
            )
        with torch.no_grad():
            val_loss, val_acc = _test_one_episode(
                supernet,
                support_data,
                support_targets,
                query_data,
                query_targets,
                args,
                path=path,
                prefix=prefix,
            )
        val_loss_avg.append(val_loss)
        val_acc_avg.append(val_acc)
        supernet.load_state_dict(model_ckpt)
        i_episode += 1
        if i_episode >= args.num_episodes:
            break
    return np.mean(val_loss_avg), np.mean(val_acc_avg)


def test(supernet, data_loader, args, path=None):
    model_ckpt = copy.deepcopy(supernet.state_dict())
    test_accs = []
    i_episode = 1
    for support_data, support_targets, query_data, query_targets in data_loader:
        support_data, support_targets, query_data, query_targets = _prepare_data(
            support_data, support_targets, query_data, query_targets, args.device
        )
        _optimizer = _init_optimizer(supernet, args.optimizer, {"lr": args.lr, "weight_decay": args.weight_decay})
        if "vit" in args.backbone:
            prefix = _get_init_prefix(supernet, support_data, support_targets)
        else:
            prefix = None
        best_train_acc = 0.0
        selected_path_idx = 99
        if path is not None and path.shape[0] > 1:
            for i_path in range(path.shape[0]):
                for j_epoch in range(args.num_epochs):
                    train_loss, train_acc = _train_one_episode(
                        supernet,
                        _optimizer,
                        support_data,
                        support_targets,
                        support_data,
                        support_targets,
                        args,
                        path=path[i_path],
                        prefix=prefix,
                    )
                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                    selected_path_idx = i_path
            selected_path = path[selected_path_idx]
            print(f"Episode: {i_episode} | Selected path: {selected_path_idx + 1}")
        else:
            for j_epoch in range(args.num_epochs):
                train_loss, train_acc = _train_one_episode(
                    supernet,
                    _optimizer,
                    support_data,
                    support_targets,
                    support_data,
                    support_targets,
                    args,
                    path=path,
                    prefix=prefix,
                )
            selected_path = path
        with torch.no_grad():
            test_loss, test_acc = _test_one_episode(
                supernet,
                support_data,
                support_targets,
                query_data,
                query_targets,
                args,
                path=selected_path,
                prefix=prefix,
            )
        test_accs.append(test_acc)
        supernet.load_state_dict(model_ckpt)
        i_episode += 1
        if i_episode >= args.num_episodes:
            break
    return test_accs


def _train_one_episode(
    supernet, optimizer, support_data, support_targets, query_data, query_targets, args, path=None, prefix=None
):
    supernet.train()
    supernet.sample_path(path)
    optimizer.zero_grad(set_to_none=True)
    if "vit" in args.backbone:
        logits, distillation_loss = supernet(support_data, support_targets, query_data, query_targets, prefix=prefix)
    elif "resnet" in args.backbone:
        logits = supernet(support_data, support_targets, query_data, query_targets)
        distillation_loss = None
    else:
        raise ValueError("Invalid backbone.")
    loss = F.cross_entropy(logits, query_targets)
    if distillation_loss is not None:
        loss += distillation_loss
    loss.backward()
    optimizer.step()
    accuracy = (logits.argmax(-1) == query_targets).sum().item() / query_data.shape[0]
    return loss.item(), accuracy


def _test_one_episode(supernet, support_data, support_targets, query_data, query_targets, args, path=None, prefix=None):
    supernet.eval()
    supernet.sample_path(path)
    if "vit" in args.backbone:
        logits, distillation_loss = supernet(support_data, support_targets, query_data, query_targets, prefix=prefix)
    elif "resnet" in args.backbone:
        logits = supernet(support_data, support_targets, query_data, query_targets)
        distillation_loss = None
    else:
        raise ValueError("Invalid backbone.")
    loss = F.cross_entropy(logits, query_targets)
    if distillation_loss is not None:
        loss += distillation_loss
    accuracy = (logits.argmax(-1) == query_targets).sum().item() / query_data.shape[0]
    return loss.item(), accuracy


def _compute_prototypes(embeddings, targets, n_way):
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith("1.1"):
            prots[i] = embeddings[(targets == i).nonzero(), :].mean(0)
        else:
            prots[i] = embeddings[(targets == i).nonzero(as_tuple=False), :].mean(0)
    return prots


def _get_init_prefix(supernet, support_data, support_targets):
    n_way = torch.unique(support_targets).shape[0]
    with torch.no_grad():
        patch_embed = supernet.model.get_patch_embed(support_data, n_way)
        prototypes = _compute_prototypes(patch_embed, support_targets, n_way)
    return prototypes


def _prepare_data(support_data, support_targets, query_data, query_targets, device):
    if support_data.dim() == 5 and support_data.shape[0] == 1:
        support_data = support_data.squeeze()
        support_targets = support_targets.squeeze()
        query_data = query_data.squeeze()
        query_targets = query_targets.squeeze()
    support_data = support_data.to(device)
    support_targets = support_targets.to(device)
    query_data = query_data.to(device)
    query_targets = query_targets.to(device)
    return support_data, support_targets, query_data, query_targets


def _init_optimizer(model, optimizer, optimizer_args):
    if optimizer not in ("adadelta", "adam", "adamw", "sgd"):
        raise ValueError("Supported optimizers: [ adadelta | adam | adamw | sgd ]")
    assert optimizer_args is not None
    _optimizer = _OPTIMIZER_LOOKUP[optimizer](model.parameters(), **optimizer_args)
    return _optimizer


def _episode_summary(episode, window_len, train_loss, train_acc, interval):
    s = f"Episode: {episode:05}\n----------\n"
    s += f"Mean train loss: {train_loss:.3f}, Mean train acc.: {train_acc:.3f}\n"
    s += f"Averaged over the {window_len} most recent episodes.\n"
    s += f"Finished in {interval:.1f} minutes."
    return s
