import argparse
import pickle

import numpy as np

from nfts.data_loaders.meta_album import MetaAlbumEpisodicDataLoader
from nfts.data_loaders.meta_dataset import DomainHandler, MetaDatasetEpisodicDataLoader
from nfts.models import model_factory
from nfts.pipelines.evaluation import meta_album_evaluation, meta_dataset_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="meta_dataset")
    parser.add_argument("--backbone", type=str, default="tsa_resnet18")
    parser.add_argument("--initialization", type=str, choices=["dino", "url"], default="url")
    parser.add_argument("--setting", type=str, choices=["sdl", "mdl"], default="sdl")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--optimizer", type=str, choices=["adadelta", "adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--target_domain", type=str, default="")
    args = parser.parse_args()

    with open(f"./models/nfts/{args.backbone}_{args.setting}.pickle", "rb") as f:
        topk_paths = pickle.load(f)
    print(
        f"Meta-test:"
        f"  Backbone: {args.backbone} ({args.initialization} pre-trained)."
        f"  Setting: {args.setting}."
        f"  Path: {str(topk_paths)}"
    )
    model = model_factory(args.backbone, args.initialization, args.setting)
    model.to(args.device)
    mean_accs = []
    sems = []
    if args.dataset == "meta_dataset":
        # for domain in DomainHandler.get_test_domains():
        # "cifar10",
        # "cifar100",
        # "mnist",
        # "mscoco",
        # "traffic_sign",
        for domain in [args.target_domain]:
            data_loader = MetaDatasetEpisodicDataLoader(args.data_path, args.setting, "test", domain=domain)
            mean_acc, sem = meta_dataset_evaluation(model, data_loader, topk_paths, args)
            mean_accs.append(mean_acc)
            sems.append(sem)
            print(f"{domain:16s} mean: {mean_acc:.3f} +- {sem:.3f}")
        print(f"all mean: {np.mean(mean_accs):.3f} +- {np.mean(sems):.3f}")
    elif args.dataset == "meta_album":
        test_episodes_config = {
            "n_way": 5,
            "min_ways": None,
            "max_ways": None,
            "k_shot": 5,
            "min_shots": 1,
            "max_shots": 20,
            "query_size": 16,
        }
        # test_datasets = "PLT_VIL,RESISC,SPT,TEX"  # Set0
        test_datasets = "BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX"  # Set0,1,2
        data_loader = MetaAlbumEpisodicDataLoader(
            args.data_path, test_datasets, args.num_episodes, test_episodes_config
        )
        mean_acc, sem = meta_album_evaluation(model, data_loader, topk_paths, args)
        print(f"mean: {mean_acc:.3f} +- {sem:.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
