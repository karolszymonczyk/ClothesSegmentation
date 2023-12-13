import torch
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict
import torch.utils.data as data
import pytorch_lightning as pl
from src.data.dataloader_factory import create_dataloader
from src.data.datasets.dataset_factory import create_dataset


class DefaultPlModule(pl.LightningModule, ABC):
    def __init__(
        self,
        criterion: Callable,
        dataset_config: Dict[str, Any],
        dataloaders_config: Dict[str, Any],
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.criterion = criterion
        self.dataset_config = dataset_config
        self.dataloaders_config = dataloaders_config
        self.lr = lr

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def step_with_metrics(
        self, batch: Dict[str, torch.Tensor], prefix: str
    ) -> Dict[str, Any]:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.lr))
        return optimizer

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        metrics = self.step_with_metrics(batch, prefix="val")
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        metrics = self.step_with_metrics(batch, prefix="test")
        self.log_dict(metrics)
        return metrics

    def train_dataloader(self) -> data.DataLoader:
        self.train_dataset = create_dataset(**self.dataset_config, is_train=True)
        train_dataloader_config = self.dataloaders_config["train"]
        train_dataloader = create_dataloader(
            self.train_dataset, **train_dataloader_config
        )
        return train_dataloader

    def val_dataloader(self) -> data.DataLoader:
        self.val_dataset = create_dataset(**self.dataset_config, is_train=False)
        val_dataloader_config = self.dataloaders_config["val"]
        val_dataloader = create_dataloader(self.val_dataset, **val_dataloader_config)
        return val_dataloader

    def test_dataloader(self) -> data.DataLoader:
        self.test_dataset = create_dataset(**self.dataset_config, is_train=False)
        test_dataloader_config = self.dataloaders_config["test"]
        test_dataloader = create_dataloader(self.test_dataset, **test_dataloader_config)
        return test_dataloader

    @staticmethod
    def _add_prefix_to_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}
