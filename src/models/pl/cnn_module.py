from typing import Any, Dict
import torch

from src.models.pl.default_pl_module import DefaultPlModule
from src.models.cnn import CNN


class CNNModule(DefaultPlModule):
    def __init__(
        self,
        dataset_config: Dict[str, Any],
        dataloaders_config: Dict[str, Any],
        lr: float = 1e-3,
    ) -> None:
        criterion = torch.nn.BCELoss()
        super().__init__(criterion, dataset_config, dataloaders_config, lr)

        self.criterion = criterion
        self.model = CNN()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        images, masks = batch["image"], batch["mask"]
        predictions = self.forward(images)
        loss = self.criterion(predictions, masks)
        self.log("train_loss", loss)
        return loss

    def step_with_metrics(
        self, batch: Dict[str, torch.Tensor], prefix: str
    ) -> Dict[str, Any]:
        images, masks = batch["image"], batch["mask"]
        predictions = self.forward(images)
        loss = self.criterion(predictions, masks)
        self.log(f"{prefix}_loss", loss)
        metrics = self._calculate_metrics(predictions, masks)
        metrics_with_prefix = self._add_prefix_to_metrics(metrics, prefix)
        self.log_dict(metrics_with_prefix)

    def _calculate_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        # TODO: implement segmentation metrics IoU and Dice Coeff
        metrics = {"iou": 0.5, "dice coeff": 0.5}
        return metrics

    def calc_accuracy(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        predicted_lables = torch.argmax(predictions, dim=1)
        accuracy = torch.sum(predicted_lables == labels)
        return accuracy / predictions.shape[0]
