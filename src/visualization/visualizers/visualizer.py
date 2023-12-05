import torch
from abc import ABC, abstractmethod


class Visualizer(ABC):
    def log_visualizations(
        self, images: torch.Tensor, predictions: torch.Tensor, epoch: int
    ) -> None:
        for image_idx, (image, prediction) in enumerate(zip(images, predictions)):
            self._log_visualization(image, prediction, image_idx, epoch)

    @abstractmethod
    def _log_visualization(
        self, image: torch.Tensor, prediction: torch.Tensor, image_idx: int, epoch: int
    ) -> None:
        pass
