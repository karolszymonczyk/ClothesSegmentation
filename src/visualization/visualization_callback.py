import torch
from typing import List
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer

from src.models.pl.default_pl_module import DefaultPlModule
from src.visualization.visualizers.visualizer import Visualizer


class VisualizationCallback(Callback):
    def __init__(self, frames: List[int], visualizer: Visualizer):
        self.frames = frames
        self.visualizer = visualizer

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: DefaultPlModule,
    ) -> None:
        if trainer.is_global_zero:
            self.log_visualization(pl_module, dataset=pl_module.val_dataset)

    def on_test_epoch_start(
        self,
        trainer: Trainer,
        pl_module: DefaultPlModule,
    ) -> None:
        if trainer.is_global_zero:
            self.log_visualization(pl_module, dataset=pl_module.test_dataset)

    def log_visualization(self, pl_module: DefaultPlModule, dataset: Dataset) -> None:
        images = [dataset[frame]["image"].to(pl_module.device) for frame in self.frames]
        masks = [dataset[frame]["mask"].to(pl_module.device) for frame in self.frames]
        stacked_images = torch.stack(images)
        stacked_masks = torch.stack(masks)
        with torch.no_grad():
            predictions = pl_module.forward(stacked_images)
        self.visualizer.log_visualizations(
            images=stacked_images.cpu(),
            predictions=stacked_masks.cpu(),
            # TODO: make model for predictoing mas
            # predictions=predictions.cpu(),
            epoch=pl_module.current_epoch,
        )
