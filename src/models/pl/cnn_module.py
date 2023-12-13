import torch
from typing import Any, Dict

from src.metrics.metrics import calc_dice_coeff, calc_iou
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

        self.mask_threshold = 0.8
        self.criterion = criterion
        self.model = CNN()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        images, masks = batch["image"], batch["mask"]
        predictions = self.forward(images)
        loss = self.criterion(predictions, masks)
        self.log("train_loss", loss)
        metrics = self.step_with_metrics(batch, prefix="train")
        self.log_dict(metrics)
        return loss

    def step_with_metrics(
        self, batch: Dict[str, torch.Tensor], prefix: str
    ) -> Dict[str, Any]:
        images, masks = batch["image"], batch["mask"]
        predictions = self.forward(images)
        loss = self.criterion(predictions, masks)
        self.log(f"{prefix}_loss", loss)
        metrics = self._calculate_metrics(predictions.detach(), masks)
        metrics_with_prefix = self._add_prefix_to_metrics(metrics, prefix)
        return metrics_with_prefix

    def _calculate_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        metrics = {
            "iou": calc_iou(predictions, labels, self.mask_threshold),
            "dice_coeff": calc_dice_coeff(predictions, labels, self.mask_threshold),
        }
        return metrics
