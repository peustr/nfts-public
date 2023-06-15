import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DomainHandler:
    @staticmethod
    def get_train_domains(setting):
        if setting == "sdl":
            return ["ilsvrc_2012"]
        if setting == "mdl":
            return [
                "aircraft",
                "cu_birds",
                "dtd",
                "fungi",
                "ilsvrc_2012",
                "omniglot",
                "quickdraw",
                "vgg_flower",
            ]

    @staticmethod
    def get_test_domains():
        return [
            "aircraft",
            "cu_birds",
            "dtd",
            "fungi",
            "ilsvrc_2012",
            "omniglot",
            "quickdraw",
            "vgg_flower",
            "cifar10",
            "cifar100",
            "mnist",
            "mscoco",
            "traffic_sign",
        ]


class MetaDatasetEpisodicDataset(Dataset):
    def __init__(self, data_path, setting, split, domain=None):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError("Supported splits: [ train | val | test ]")
        if split == "test" and domain is None:
            raise ValueError("A test domain must be specified during testing.")
        self.data_path = data_path
        self.setting = setting
        self.split = split
        self.domain = domain
        self._root_dir = os.path.join(self.data_path, "cached", self.split)
        if self.split in ("train", "val"):
            self._root_dir = os.path.join(self._root_dir, self.setting)
        elif self.split == "test":
            self._root_dir = os.path.join(self._root_dir, self.domain)
        self._num_episodes = len([ptfile for ptfile in os.listdir(self._root_dir) if ptfile.endswith(".pt")]) // 4
        self.filenames = [f"E{index:05}" for index in range(1, self._num_episodes + 1)]
        np.random.shuffle(self.filenames)
        print(self.filenames[:10])

    def __getitem__(self, index):
        support_data = torch.load(self._episode_filepath(self.filenames[index], "SX"))
        support_targets = torch.load(self._episode_filepath(self.filenames[index], "SY"))
        query_data = torch.load(self._episode_filepath(self.filenames[index], "QX"))
        query_targets = torch.load(self._episode_filepath(self.filenames[index], "QY"))
        return support_data, support_targets, query_data, query_targets

    def __len__(self):
        return self._num_episodes

    def _episode_filepath(self, filename, id_):
        return os.path.join(self._root_dir, f"{filename}{id_}.pt")


def MetaDatasetEpisodicDataLoader(data_path, setting, split, domain=None):
    return DataLoader(
        MetaDatasetEpisodicDataset(data_path, setting, split, domain=domain),
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
