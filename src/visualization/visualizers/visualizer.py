import torch
import numpy as np
from os.path import join
from abc import ABC, abstractmethod
from torchvision.utils import save_image


class Visualizer(ABC):
    def _save_image(self, image: np.ndarray, image_idx: int, epoch: int) -> None:
        name = f"{self.prefix_name}_{image_idx:02d}"
        file_name = join(self.save_path, f"{name}_epoch_{epoch}.png")
        save_image(image, file_name)
        if self.logger is not None:
            self.logger.log_image(name, file_name, epoch)

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
