import argparse
import os
import resource

import torch

from nfts.data_loaders.meta_dataset_tf import (
    DomainHandler,
    TF_MetaDatasetEpisodicDataLoader,
)
from nfts.tfutils import disable_tf_gpu_access, limit_tf_cpu_usage


def main(args):
    def _episode_filepath(root_dir, episode, id_):
        return os.path.join(root_dir, f"E{episode:05}{id_}.pt")

    root_dir = os.path.join(args.data_path, "cached", args.split)
    if args.split in ("train", "val"):
        setting_dir = os.path.join(root_dir, args.setting)
        os.makedirs(setting_dir, exist_ok=True)
        data_loader = TF_MetaDatasetEpisodicDataLoader(args.data_path, args.gin_path, args.setting, args.split, "cpu")
        for i_episode in range(1, args.num_episodes + 1):
            support_data, support_targets, query_data, query_targets, domain_id = data_loader.fetch()
            torch.save(support_data, _episode_filepath(setting_dir, i_episode, "SX"))
            torch.save(support_targets, _episode_filepath(setting_dir, i_episode, "SY"))
            torch.save(query_data, _episode_filepath(setting_dir, i_episode, "QX"))
            torch.save(query_targets, _episode_filepath(setting_dir, i_episode, "QY"))
            print(f"E{i_episode:05}, D{domain_id:02}", support_data.shape, query_data.shape)
    elif args.split == "test":
        domains = DomainHandler.get_test_domains()
        for domain in domains:
            domain_dir = os.path.join(root_dir, domain)
            os.makedirs(domain_dir, exist_ok=True)
            data_loader = TF_MetaDatasetEpisodicDataLoader(
                args.data_path, args.gin_path, args.setting, args.split, "cpu", domain=domain
            )
            for i_episode in range(1, args.num_episodes + 1):
                support_data, support_targets, query_data, query_targets, domain_id = data_loader.fetch()
                torch.save(support_data, _episode_filepath(domain_dir, i_episode, "SX"))
                torch.save(support_targets, _episode_filepath(domain_dir, i_episode, "SY"))
                torch.save(query_data, _episode_filepath(domain_dir, i_episode, "QX"))
                torch.save(query_targets, _episode_filepath(domain_dir, i_episode, "QY"))
                print(f"E{i_episode:05}, D{domain}", support_data.shape, query_data.shape)


if __name__ == "__main__":
    disable_tf_gpu_access()
    limit_tf_cpu_usage()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--gin_path", type=str, default="./gin_files/data_config.gin")
    parser.add_argument("--setting", type=str, choices=["sdl", "mdl"], default="sdl")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--num_episodes", type=int, default=50000)
    args = parser.parse_args()
    if args.setting == "mdl":
        resource.setrlimit(resource.RLIMIT_NOFILE, (10000, 10000))
    main(args)
