from PIL import Image
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint


class MLFlowLogger(loggers.MLFlowLogger):
    def log_file(self, path: str, artifact_path: str) -> None:
        self.experiment.log_artifact(self.run_id, path, artifact_path=artifact_path)

    def log_image(self, name: str, image_path: str, epoch: int) -> None:
        self.experiment.log_image(
            self.run_id, Image.open(image_path), f"epoch_{epoch:02d}_{name}.png"
        )

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        self.experiment.log_artifact(
            self.run_id,
            checkpoint_callback.best_model_path,
            artifact_path="model_checkpoint",
        )
