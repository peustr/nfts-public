import os

import gin
import torch
from meta_dataset.data.config import EpisodeDescriptionConfig
from meta_dataset.data.dataset_spec import load_dataset_spec
from meta_dataset.data.learning_spec import Split
from meta_dataset.data.pipeline import (
    make_multisource_episode_pipeline,
    make_one_source_episode_pipeline,
)
from torchvision import transforms as T

_DATA_SPLIT_LOOKUP = {"train": Split.TRAIN, "val": Split.VALID, "test": Split.TEST}


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize():
    return T.Normalize(IMAGENET_MEAN, IMAGENET_STD)


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


class TF_MetaDatasetEpisodicDataLoader:
    def __init__(self, data_path, gin_path, setting, split, device, domain=None):
        if split not in ("train", "val", "test"):
            raise ValueError("Supported splits: [ train | val | test ]")
        if split == "test" and domain is None:
            raise ValueError("A test domain must be specified during testing.")
        self.data_path = data_path
        self.gin_path = gin_path
        self.setting = setting
        self.split = split
        if self.split == "test":
            self.domains = [domain]
        else:
            if domain is not None:
                self.domains = [domain]
            else:
                self.domains = DomainHandler.get_train_domains(self.setting)
        self.device = device
        self.domain = domain
        self.transforms = imagenet_normalize()
        gin.parse_config_file(self.gin_path)
        self._episodic_data_iterator = self._create_data_iterator()

    def _create_data_iterator(self):
        num_domains = len(self.domains)
        dataset_spec_list = [load_dataset_spec(os.path.join(self.data_path, dataset_id)) for dataset_id in self.domains]
        use_bilevel_ontology_list = [False] * num_domains
        use_dag_ontology_list = [False] * num_domains
        if "ilsvrc_2012" in self.domains:
            use_dag_ontology_list[self.domains.index("ilsvrc_2012")] = True
        if "omniglot" in self.domains:
            use_bilevel_ontology_list[self.domains.index("omniglot")] = True
        episode_descr_config = EpisodeDescriptionConfig(num_ways=None, num_support=None, num_query=None)
        if num_domains == 1:
            episodic_data_iterator = make_one_source_episode_pipeline(
                dataset_spec=dataset_spec_list[0],
                use_dag_ontology=use_dag_ontology_list[0],
                use_bilevel_ontology=use_bilevel_ontology_list[0],
                episode_descr_config=episode_descr_config,
                split=_DATA_SPLIT_LOOKUP[self.split],
                image_size=224,
                shuffle_buffer_size=1024,
            ).as_numpy_iterator()
        else:
            episodic_data_iterator = make_multisource_episode_pipeline(
                dataset_spec_list=dataset_spec_list,
                use_dag_ontology_list=use_dag_ontology_list,
                use_bilevel_ontology_list=use_bilevel_ontology_list,
                episode_descr_config=episode_descr_config,
                split=_DATA_SPLIT_LOOKUP[self.split],
                image_size=224,
                shuffle_buffer_size=1024,
            ).as_numpy_iterator()
        return episodic_data_iterator

    def fetch(self):
        episode, domain_id = next(self._episodic_data_iterator)
        sx, sy, _, qx, qy, _ = episode
        support_data, support_targets, query_data, query_targets = (
            self.transforms(_preprocess_np_images(sx)).to(self.device),
            _preprocess_np_labels(sy).to(self.device),
            self.transforms(_preprocess_np_images(qx)).to(self.device),
            _preprocess_np_labels(qy).to(self.device),
        )
        return support_data, support_targets, query_data, query_targets, domain_id


def _preprocess_np_images(np_img):
    return torch.from_numpy(np_img.transpose((0, 3, 1, 2)))


def _preprocess_np_labels(np_lbl):
    return torch.from_numpy(np_lbl).long()
