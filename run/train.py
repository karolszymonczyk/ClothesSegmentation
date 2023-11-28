import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils import create_mlp_module


CONFIG = {
    "max_epochs": 1,
    "experiment_name": "setup",
    "run_name": "test_run",
    "checkpoint_dir": "mlcheckpoints",
    "tracking_uri": "sqlite:///mlflow.db",
}

MODEL_CONFIG = {
    "input_size": 28 * 28,
    "hidden_size": 128,
    "output_size": 10,
    "lr": 1e-3,
}


def train() -> None:
    mlf_logger = MLFlowLogger(
        experiment_name=CONFIG["experiment_name"],
        run_name=CONFIG["run_name"],
        tracking_uri=CONFIG["tracking_uri"],
    )

    model = create_mlp_module(**MODEL_CONFIG)
    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        logger=mlf_logger,
        callbacks=[
            ModelCheckpoint(
                CONFIG["checkpoint_dir"],
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            )
        ],
    )
    trainer.fit(model)


if __name__ == "__main__":
    train()
