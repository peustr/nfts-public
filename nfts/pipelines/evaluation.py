import numpy as np

from nfts.pipelines.core import test


def meta_dataset_evaluation(model, data_loader, topk_paths, args):
    accs = test(model, data_loader, args, path=topk_paths)
    mean_acc = np.mean(accs)
    sem = np.std(accs) / np.sqrt(len(accs))
    return mean_acc, sem


def meta_album_evaluation(model, data_loader, topk_paths, args):
    accs = test(model, data_loader, args, path=topk_paths)
    mean_acc = np.mean(accs)
    sem = np.std(accs) / np.sqrt(len(accs))
    return mean_acc, sem
