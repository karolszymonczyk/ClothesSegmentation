import yaml
from pathlib import Path
from typing import Dict, Union


def load_config(config_path: Union[str, Path]) -> Dict:
    with open(config_path, "r") as config_file:
        return yaml.load(config_file, Loader=yaml.FullLoader)
