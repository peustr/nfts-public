import argparse

import torch

from nfts.data_loaders.meta_album import MetaAlbumEpisodicDataLoader
from nfts.data_loaders.meta_dataset import MetaDatasetEpisodicDataLoader
from nfts.models import model_factory
from nfts.pipelines.core import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="meta_dataset")
    parser.add_argument("--backbone", type=str, default="tsa_resnet18")
    parser.add_argument("--initialization", type=str, choices=["dino", "url"], default="url")
    parser.add_argument("--setting", type=str, choices=["sdl", "mdl"], default="sdl")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=50000)
    parser.add_argument("--optimizer", type=str, choices=["adadelta", "adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    print(
        f"Supernet Training:"
        f"  Backbone: {args.backbone} ({args.initialization} pre-trained)."
        f"  Setting: {args.setting}."
    )
    if args.dataset == "meta_dataset":
        data_loader = MetaDatasetEpisodicDataLoader(args.data_path, args.setting, "train")
    elif args.dataset == "meta_album":
        train_episodes_config = {
            "n_way": 5,
            "min_ways": None,
            "max_ways": None,
            "k_shot": None,
            "min_shots": 1,
            "max_shots": 20,
            "query_size": 16,
        }
        # train_datasets = "BCT,BRD,CRS"  # Set0
        train_datasets = "DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS"  # Set0,1,2
        data_loader = MetaAlbumEpisodicDataLoader(
            args.data_path, train_datasets, args.num_episodes, train_episodes_config
        )
    model = model_factory(args.backbone, args.initialization, args.setting)
    model.to(args.device)
    train(model, data_loader, args)
    torch.save(model.state_dict(), f"./models/nfts/{args.backbone}_{args.setting}.pth")
    print("Done.")


if __name__ == "__main__":
    main()
