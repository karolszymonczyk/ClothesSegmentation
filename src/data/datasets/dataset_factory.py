from typing import Optional, Tuple
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset

from src.data.datasets.deep_fashion_mask_dataset import DeepFashionMaskDataset


def create_fashionmnist_dataset(root_path: str, is_train: bool) -> Dataset:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.FashionMNIST(
        root=root_path, train=is_train, download=True, transform=transform
    )
    return dataset


def create_deep_fashion_mask_dataset(
    root_path: str, is_train: bool, target_size_hw: Tuple[int, int]
) -> Dataset:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # already normalize images to [0, 1] range
            transforms.Resize(target_size_hw, antialias=True),
        ]
    )
    dataset = DeepFashionMaskDataset(root_path, is_train, target_size_hw, transform)
    return dataset


DATASETS_CREATORS = {
    "FashionMNIST": create_fashionmnist_dataset,
    "DeepFashionMask": create_deep_fashion_mask_dataset,
}


def create_dataset(
    name: str, subset: Optional[int] = None, **dataset_config
) -> Dataset:
    dataset_creator = DATASETS_CREATORS[name]
    dataset = dataset_creator(**dataset_config)

    if subset is not None:
        dataset = _make_into_subset(dataset, subset)

    return dataset


def _make_into_subset(dataset: Dataset, subset_size: int) -> Subset:
    subset_indices = list(range(subset_size))
    return Subset(dataset, indices=subset_indices)
