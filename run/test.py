import argparse
import pytorch_lightning as pl

from src.utils import create_mlp_module, load_config


def test(args: argparse.Namespace) -> None:
    config = load_config(args.config_path)
    model_config = config["model"]

    model = create_mlp_module(**model_config)
    trainer = pl.Trainer()
    trainer.test(model, verbose=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run this script to test model created from given config."
    )
    parser.add_argument(
        "-c", "--config_path", help="Path to config file", required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    test(args)
