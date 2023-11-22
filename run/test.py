import pytorch_lightning as pl

from src.utils import create_mlp_module


CONFIG = {"checkpoint_path": "./models/mlp.pth"}

MODEL_CONFIG = {
    "input_size": 28 * 28,
    "hidden_size": 128,
    "output_size": 10,
    "lr": 1e-3,
}


def test() -> None:
    model = create_mlp_module(checkpoint_path=CONFIG["checkpoint_path"], **MODEL_CONFIG)
    trainer = pl.Trainer()
    trainer.test(model, verbose=True)


if __name__ == "__main__":
    test()
