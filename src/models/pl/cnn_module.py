from typing import Any, Dict, Tuple
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
        self.log_dict(metrics_with_prefix)

    def _calculate_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        metrics = {
            "iou": self.calc_iou(predictions, labels, self.mask_threshold),
            "dice_coeff": self.calc_dice_coeff(
                predictions, labels, self.mask_threshold
            ),
        }
        return metrics

    def calc_iou(
        self, predictions: torch.Tensor, labels: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        intersection, union = self.calc_intersection_and_union_per_sample(
            predictions, labels, threshold
        )
        iou = intersection / union
        return iou.mean()

    def calc_dice_coeff(
        self, predictions: torch.Tensor, labels: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        intersection, union = self.calc_intersection_and_union_per_sample(
            predictions, labels, threshold
        )
        dice_coefficient = (2.0 * intersection) / (union + intersection)
        return dice_coefficient.mean()

    def calc_intersection_and_union_per_sample(
        self, mask1: torch.Tensor, mask2: torch.Tensor, threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        binary_mask1 = mask1 >= threshold
        binary_mask2 = mask2 >= threshold

        intersection = torch.logical_and(binary_mask1, binary_mask2)
        union = torch.logical_or(binary_mask1, binary_mask2)

        batch_size = intersection.shape[0]
        intersection_per_sample = intersection.view(batch_size, -1)
        union_per_sample = union.view(batch_size, -1)

        intersection_sum = torch.sum(intersection_per_sample, dim=1)
        union_sum = torch.sum(union_per_sample, dim=1)

        return intersection_sum, union_sum
