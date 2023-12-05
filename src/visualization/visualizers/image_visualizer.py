import torch
import numpy as np
from typing import Tuple
from os.path import join
from torchvision import transforms
from torchvision.utils import save_image
from src.utils import crate_if_not_exists

from src.visualization.visualizers.visualizer import Visualizer


class ImageVisualizer(Visualizer):
    def __init__(self, logger: int, save_path: str, image_size_hw: Tuple[int, int]):
        self.logger = logger
        self.save_path = save_path
        crate_if_not_exists(self.save_path)
        self.image_size_hw = image_size_hw
        self.prefix_name = "image"

    def _log_visualization(
        self, image: torch.Tensor, prediction: torch.Tensor, image_idx: int, epoch: int
    ) -> None:
        resize = transforms.Resize(self.image_size_hw, antialias=True)
        resized_image = resize(image.cpu())
        self._save_image(resized_image, image_idx, epoch)

    def _save_image(self, image: np.ndarray, image_idx: int, epoch: int) -> None:
        name = f"{self.prefix_name}_{image_idx:02d}"
        file_name = join(self.save_path, f"{name}_epoch_{epoch}.png")
        save_image(image, file_name)
        if self.logger is not None:
            self.logger.log_image(name, file_name, epoch)
