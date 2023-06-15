import copy
from collections import deque
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adadelta, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from nfts.models.attack import pgd
from nfts.pipelines.utils import cosine_scheduler

_OPTIMIZER_LOOKUP = {
    "adadelta": Adadelta,
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
}


def train(model, data_loader, args, use_adversarial_loss=False):
    if args.num_epochs != 1:
        raise ValueError("Number of epochs should be 1 during training.")
    if args.num_episodes >= 1000:
        window_len = 1000
    else:
        window_len = int(np.clip(args.num_episodes // 10, 1, 100))
    _optimizer = _init_optimizer(model, args.optimizer, {"lr": args.lr, "weight_decay": args.weight_decay})
    if args.optimizer == "sgd":
        scheduler = CosineAnnealingWarmRestarts(_optimizer, T_0=args.num_episodes // 5)
    else:
        scheduler = None
    if args.optimizer == "adamw" and args.num_warmup_episodes > 0:
        lr_scheduler = cosine_scheduler(args.lr, args.lr * 0.1, args.num_episodes, 1, args.num_warmup_episodes)
        wd_scheduler = cosine_scheduler(args.weight_decay, args.weight_decay * 10.0, args.num_episodes, 1)
        use_warmup = True
    else:
        use_warmup = False
    train_loss_avg, train_acc_avg = deque(maxlen=window_len), deque(maxlen=window_len)
    i_episode = 1
    for support_data, support_targets, query_data, query_targets in data_loader:
        support_data = support_data.to(args.device)[0]
        support_targets = support_targets.to(args.device)[0]
        query_data = query_data.to(args.device)[0]
        query_targets = query_targets.to(args.device)[0]
        start_time = timer()
        for j_epoch in range(args.num_epochs):
            if use_warmup:
                for i, param_group in enumerate(_optimizer.param_groups):
                    param_group["lr"] = lr_scheduler[i_episode - 1]
                    if i == 0:
                        param_group["weight_decay"] = wd_scheduler[i_episode - 1]
            if use_adversarial_loss:
                adv_support_data = pgd(
                    model, support_data, support_targets, query_data, query_targets, epsilon=16.0 / 255.0
                )
                train_loss, train_acc = _train_one_episode_adversarial(
                    model,
                    _optimizer,
                    support_data,
                    adv_support_data,
                    support_targets,
                    query_data,
                    query_targets,
                )
            else:
                train_loss, train_acc = _train_one_episode(
                    model,
                    _optimizer,
                    support_data,
                    support_targets,
                    query_data,
                    query_targets,
                )
        if scheduler is not None:
            scheduler.step()
        train_loss_avg.append(train_loss)
        train_acc_avg.append(train_acc)
        end_time = timer()
        interval = (end_time - start_time) / 60.0
        print(_episode_summary(i_episode, window_len, np.mean(train_loss_avg), np.mean(train_acc_avg), interval))
        i_episode += 1
    return np.mean(train_loss_avg), np.mean(train_acc_avg)


def validate(model, data_loader, args):
    model_ckpt = copy.deepcopy(model.state_dict())
    val_loss_avg, val_acc_avg = [], []
    for support_data, support_targets, query_data, query_targets in data_loader:
        support_data = support_data.to(args.device)[0]
        support_targets = support_targets.to(args.device)[0]
        query_data = query_data.to(args.device)[0]
        query_targets = query_targets.to(args.device)[0]
        _optimizer = _init_optimizer(model, args.optimizer, {"lr": args.lr, "weight_decay": args.weight_decay})
        for j_epoch in range(args.num_epochs):
            train_loss, train_acc = _train_one_episode(
                model,
                _optimizer,
                support_data,
                support_targets,
                support_data,
                support_targets,
            )
        with torch.no_grad():
            val_loss, val_acc = _test_one_episode(
                model,
                support_data,
                support_targets,
                query_data,
                query_targets,
            )
        val_loss_avg.append(val_loss)
        val_acc_avg.append(val_acc)
        model.load_state_dict(model_ckpt)
    return np.mean(val_loss_avg), np.mean(val_acc_avg)


def test(model, data_loader, args):
    model_ckpt = copy.deepcopy(model.state_dict())
    test_accs = []
    for support_data, support_targets, query_data, query_targets in data_loader:
        support_data = support_data.to(args.device)[0]
        support_targets = support_targets.to(args.device)[0]
        query_data = query_data.to(args.device)[0]
        query_targets = query_targets.to(args.device)[0]
        _optimizer = _init_optimizer(model, args.optimizer, {"lr": args.lr, "weight_decay": args.weight_decay})
        for j_epoch in range(args.num_epochs):
            train_loss, train_acc = _train_one_episode(
                model,
                _optimizer,
                support_data,
                support_targets,
                support_data,
                support_targets,
            )
        with torch.no_grad():
            test_loss, test_acc = _test_one_episode(
                model,
                support_data,
                support_targets,
                query_data,
                query_targets,
            )
        test_accs.append(test_acc)
        model.load_state_dict(model_ckpt)
    return test_accs


def _train_one_episode(model, optimizer, support_data, support_targets, query_data, query_targets):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(support_data, support_targets, query_data, query_targets)
    loss = F.cross_entropy(logits, query_targets)
    loss.backward()
    optimizer.step()
    accuracy = (logits.argmax(-1) == query_targets).sum().item() / query_data.shape[0]
    return loss.item(), accuracy


def _train_one_episode_adversarial(
    model, optimizer, support_data, adv_support_data, support_targets, query_data, query_targets
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(support_data, support_targets, query_data, query_targets)
    adv_logits = model(adv_support_data, support_targets, query_data, query_targets)
    loss = F.cross_entropy(logits, query_targets) + F.cross_entropy(adv_logits, query_targets)
    loss.backward()
    optimizer.step()
    accuracy = (logits.argmax(-1) == query_targets).sum().item() / query_data.shape[0]
    return loss.item(), accuracy


def _test_one_episode(model, support_data, support_targets, query_data, query_targets):
    model.eval()
    logits = model(support_data, support_targets, query_data, query_targets)
    loss = F.cross_entropy(logits, query_targets)
    accuracy = (logits.argmax(-1) == query_targets).sum().item() / query_data.shape[0]
    return loss.item(), accuracy


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
