from typing import Any, Dict
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from src.data.dataloader_factory import create_dataloader
from src.data.datasets.dataset_factory import create_dataset

from src.models.mlp import MLP


class MLPModule(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dataset_config: Dict[str, Any],
        dataloaders_config: Dict[str, Any],
        lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.model = MLP(input_size, hidden_size, output_size)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr

        self.dataset_config = dataset_config
        self.dataloaders_config = dataloaders_config

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.lr))
        return optimizer

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        images, labels = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def train_dataloader(self) -> data.DataLoader:
        self.train_dataset = create_dataset(**self.dataset_config, is_train=True)
        train_dataloader_config = self.dataloaders_config["train"]
        train_dataloader = create_dataloader(
            self.train_dataset, **train_dataloader_config
        )
        return train_dataloader

    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        images, labels = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        self.log("val_loss", loss)
        metrics = self._calculate_metrics(predictions, labels)
        self.log_dict(metrics)
        return metrics

    def val_dataloader(self) -> data.DataLoader:
        self.val_dataset = create_dataset(**self.dataset_config, is_train=False)
        val_dataloader_config = self.dataloaders_config["val"]
        val_dataloader = create_dataloader(self.val_dataset, **val_dataloader_config)
        return val_dataloader

    def test_step(self, batch: torch.Tensor) -> torch.Tensor:
        images, labels = batch[0], batch[1]
        predictions = self.forward(images)
        metrics = self._calculate_metrics(predictions, labels)
        self.log_dict(metrics)
        return metrics

    def test_dataloader(self) -> data.DataLoader:
        self.test_dataset = create_dataset(**self.dataset_config, is_train=False)
        test_dataloader_config = self.dataloaders_config["test"]
        test_dataloader = create_dataloader(self.test_dataset, **test_dataloader_config)
        return test_dataloader

    def _calculate_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        metrics = {"accuracy": self.calc_accuracy(predictions, labels)}
        return metrics

    def calc_accuracy(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        predicted_lables = torch.argmax(predictions, dim=1)
        accuracy = torch.sum(predicted_lables == labels)
        return accuracy / predictions.shape[0]
