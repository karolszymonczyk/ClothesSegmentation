import shutil
import unittest
from pathlib import Path

from src.utils import crate_if_not_exists, load_config


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_data_path = "./tests/mock_data"

    def test_load_config(self):
        # given
        mock_config_path = f"{self.mock_data_path}/configs/train_config.yml"

        # when
        config = load_config(mock_config_path)

        # then
        self.assertTrue(isinstance(config, dict))

    def test_crate_if_not_exists(self):
        # given
        self.tmp_dir_path = Path(f"{self.mock_data_path}/tmp_dir")
        if Path.exists(self.tmp_dir_path):
            shutil.rmtree(self.tmp_dir_path)

        # when
        crate_if_not_exists(self.tmp_dir_path)

        # then
        self.assertTrue(Path.exists(self.tmp_dir_path))
