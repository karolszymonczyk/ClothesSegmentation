from typing import Optional
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset


def create_fashionmnist_dataset(root_path: str, is_train: bool) -> Dataset:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.FashionMNIST(
        root=root_path, train=is_train, download=True, transform=transform
    )
    return dataset


DATASETS_CREATORS = {
    "FashionMNIST": create_fashionmnist_dataset,
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
