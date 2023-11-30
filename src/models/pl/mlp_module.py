from typing import Any, Dict
import torch

from src.models.pl.default_pl_module import DefaultPlModule
from src.models.mlp import MLP


class MLPModule(DefaultPlModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dataset_config: Dict[str, Any],
        dataloaders_config: Dict[str, Any],
        lr: float = 1e-3,
    ) -> None:
        criterion = torch.nn.CrossEntropyLoss()
        super().__init__(criterion, dataset_config, dataloaders_config, lr)

        self.criterion = criterion
        self.model = MLP(input_size, hidden_size, output_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        images, labels = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def step_with_metrics(
        self, batch: Dict[str, torch.Tensor], prefix: str
    ) -> Dict[str, Any]:
        images, labels = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        self.log(f"{prefix}_loss", loss)
        metrics = self._calculate_metrics(predictions, labels)
        metrics_with_prefix = self._add_prefix_to_metrics(metrics, prefix)
        self.log_dict(metrics_with_prefix)

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
