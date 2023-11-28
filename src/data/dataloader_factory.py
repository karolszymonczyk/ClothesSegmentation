from torch.utils.data import DataLoader, Dataset


def create_default_dataloader(dataset: Dataset, **config) -> DataLoader:
    return DataLoader(
        dataset,
        shuffle=config["shuffle"],
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        persistent_workers=config["persistent_workers"],
    )


DATALOADER_CREATORS = {
    "DefaultDataLoader": create_default_dataloader,
}


def create_dataloader(
    dataset: Dataset, name="DefaultDataLoader", **dataloader_config
) -> DataLoader:
    model_creator = DATALOADER_CREATORS[name]
    return model_creator(dataset, **dataloader_config)
