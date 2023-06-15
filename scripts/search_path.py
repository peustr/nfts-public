import argparse
import pickle

from nfts.data_loaders.meta_album import MetaAlbumEpisodicDataLoader
from nfts.data_loaders.meta_dataset import MetaDatasetEpisodicDataLoader
from nfts.models import model_factory
from nfts.pipelines.search import evolutionary_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="meta_dataset")
    parser.add_argument("--backbone", type=str, default="tsa_resnet18")
    parser.add_argument("--initialization", type=str, choices=["dino", "url"], default="url")
    parser.add_argument("--setting", type=str, choices=["sdl", "mdl"], default="sdl")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--population_size", type=int, default=64)
    parser.add_argument("--topk_crossover", type=int, default=8)
    parser.add_argument("--max_evaluations", type=int, default=1000)
    parser.add_argument("--optimizer", type=str, choices=["adadelta", "adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    print(
        f"Supernet Path Search:"
        f"  Backbone: {args.backbone} ({args.initialization} pre-trained)."
        f"  Setting: {args.setting}."
    )
    if args.dataset == "meta_dataset":
        data_loader = MetaDatasetEpisodicDataLoader(args.data_path, args.setting, "train")
    elif args.dataset == "meta_album":
        val_episodes_config = {
            "n_way": 5,
            "min_ways": None,
            "max_ways": None,
            "k_shot": 5,
            "min_shots": 1,
            "max_shots": 20,
            "query_size": 16,
        }
        # val_datasets = "FLW,MD_MIX,PLK"  # Set0
        val_datasets = "AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6"  # Set0,1,2
        data_loader = MetaAlbumEpisodicDataLoader(args.data_path, val_datasets, args.num_episodes, val_episodes_config)
    model = model_factory(args.backbone, args.initialization, args.setting)
    model.to(args.device)
    topk_paths = evolutionary_search(model, data_loader, args)
    # top_path = evolutionary_search(model, data_loader, args)
    with open(f"./models/nfts/{args.backbone}_{args.setting}.pickle", "wb") as f:
        pickle.dump(topk_paths, f, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(top_path, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")


if __name__ == "__main__":
    main()
