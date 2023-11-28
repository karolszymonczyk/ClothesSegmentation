import torch
import yaml
from pathlib import Path
from typing import Dict, List, Union

from src.models.pl.mlp_module import MLPModule


def batch_data(
    data: Union[List, torch.Tensor], batch_size: int
) -> Union[List, torch.Tensor]:
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def load_config(config_path: Union[str, Path]) -> Dict:
    with open(config_path, "r") as config_file:
        return yaml.load(config_file, Loader=yaml.FullLoader)


def create_mlp_module(checkpoint_path: str = None, **config) -> MLPModule:
    if checkpoint_path is not None:
        return MLPModule.load_from_checkpoint(checkpoint_path=checkpoint_path, **config)
    return MLPModule(**config)
