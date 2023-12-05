from typing import Any, Dict
import torch
from src.models.mock import Mock

from src.models.pl.default_pl_module import DefaultPlModule


class MockModule(DefaultPlModule):
    def __init__(
        self,
        input_dim: int,
        dataset_config: Dict[str, Any],
        dataloaders_config: Dict[str, Any],
        lr: float = 1e-3,
    ) -> None:
        criterion = torch.nn.CrossEntropyLoss()
        super().__init__(criterion, dataset_config, dataloaders_config, lr)

        self.criterion = criterion
        self.model = Mock(input_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        images = batch["image"]
        pred_img = self.forward(images)
        labels = torch.ones_like(pred_img)
        loss = self.criterion(pred_img, labels)
        self.log("train_loss", loss)
        return loss

    def step_with_metrics(
        self, batch: Dict[str, torch.Tensor], prefix: str
    ) -> Dict[str, Any]:
        images = batch["image"]
        pred_img = self.forward(images)
        labels = torch.ones_like(pred_img)
        loss = self.criterion(pred_img, labels)
        self.log(f"{prefix}_loss", loss)
        return {"val_loss": loss}
