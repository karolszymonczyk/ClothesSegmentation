import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from src.utils import create_mlp_module


CONFIG = {
    "max_epochs": 1,
    "model_save_path": "./models/mlp.pth",
}

MODEL_CONFIG = {
    "input_size": 28 * 28,
    "hidden_size": 128,
    "output_size": 10,
    "lr": 1e-3,
}


def train() -> None:
    mlf_logger = MLFlowLogger(
        experiment_name="lightning_logs", tracking_uri="sqlite:///mlflow.db"
    )

    model = create_mlp_module(**MODEL_CONFIG)
    trainer = pl.Trainer(max_epochs=CONFIG["max_epochs"], logger=mlf_logger)
    trainer.fit(model)
    trainer.save_checkpoint(CONFIG["model_save_path"])


if __name__ == "__main__":
    train()
