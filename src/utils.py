import os
import yaml
from pathlib import Path
from typing import Dict, Union


def load_config(config_path: Union[str, Path]) -> Dict:
    with open(config_path, "r") as config_file:
        return yaml.load(config_file, Loader=yaml.FullLoader)


def crate_if_not_exists(path: Union[str, Path]) -> None:
    path_object = Path(path)
    if not Path.exists(path_object):
        os.makedirs(path_object)
