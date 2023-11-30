from typing import Any, Dict
import torch
from src.models.mock import Mock

from src.models.pl.default_pl_module import DefaultPlModule


class MockModule(DefaultPlModule):
    def __init__(
        self,
        dataset_config: Dict[str, Any],
        dataloaders_config: Dict[str, Any],
        lr: float = 1e-3,
    ) -> None:
        criterion = torch.nn.CrossEntropyLoss()
        super().__init__(criterion, dataset_config, dataloaders_config, lr)

        self.criterion = criterion
        self.model = Mock()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        pass

    def step_with_metrics(
        self, batch: Dict[str, torch.Tensor], prefix: str
    ) -> Dict[str, Any]:
        pass

    def _calculate_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        pass
