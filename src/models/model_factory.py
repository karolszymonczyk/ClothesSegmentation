from torch import nn
from typing import Callable, Dict
from src.models.pl.mlp_module import MLPModule
from src.models.pl.mock_module import MockModule


def create_mlp_module(checkpoint_path: str = None, **config) -> MLPModule:
    if checkpoint_path is not None:
        return MLPModule.load_from_checkpoint(checkpoint_path=checkpoint_path, **config)
    return MLPModule(**config)


def create_mock_module(checkpoint_path: str = None, **config) -> MLPModule:
    img_heigth, img_width = config["dataset_config"]["target_size_hw"]
    input_dim = 3 * img_heigth * img_width
    return MockModule(input_dim=input_dim, **config)


MODEL_CREATORS: Dict[str, Callable] = {
    "MLP": create_mlp_module,
    "Mock": create_mock_module,
}


def create_model(name: str, **model_config) -> nn.Module:
    model_creator = MODEL_CREATORS[name]
    return model_creator(**model_config)
