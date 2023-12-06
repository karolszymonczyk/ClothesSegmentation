import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.loggers.mlflow_logger import MLFlowLogger
from src.models.model_factory import create_model
from src.utils import load_config
from src.visualization.visualization_callback import VisualizationCallback
from src.visualization.visualizers.image_visualizer import ImageVisualizer
from src.visualization.visualizers.mask_visualizer import MaskVisualizer


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config_path)
    logger_config = config["logger"]
    model_config = config["model"]
    visualization_config = config["visualization"]

    mlf_logger = MLFlowLogger(
        experiment_name=logger_config["experiment_name"],
        run_name=logger_config["run_name"],
        tracking_uri=logger_config["tracking_uri"],
    )

    model = create_model(**model_config)
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        precision=config["precision"],
        logger=mlf_logger,
        callbacks=[
            ModelCheckpoint(
                logger_config["checkpoint_dir"],
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            VisualizationCallback(
                visualizer=MaskVisualizer(
                    logger=mlf_logger,
                    save_path=visualization_config["save_path"],
                    image_size_hw=visualization_config["image_size_hw"],
                ),
                frames=visualization_config["frames"],
            ),
        ],
    )
    mlf_logger.log_file(path=args.config_path, artifact_path="train_config")
    trainer.fit(model)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run this script to train model created from given config."
    )
    parser.add_argument(
        "-c", "--config_path", help="Path to config file", required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
