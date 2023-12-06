import torch
from typing import Tuple
from torchvision import transforms
from src.utils import crate_if_not_exists

from src.visualization.visualizers.visualizer import Visualizer


class MaskVisualizer(Visualizer):
    def __init__(self, logger: int, save_path: str, image_size_hw: Tuple[int, int]):
        self.logger = logger
        self.save_path = save_path
        crate_if_not_exists(self.save_path)
        self.image_size_hw = image_size_hw
        self.prefix_name = "mask"

    def _log_visualization(
        self, image: torch.Tensor, prediction: torch.Tensor, image_idx: int, epoch: int
    ) -> None:
        image_with_mask = image * prediction
        resize = transforms.Resize(self.image_size_hw, antialias=True)
        resized_image_with_mask = resize(image_with_mask)
        self._save_image(resized_image_with_mask, image_idx, epoch)
