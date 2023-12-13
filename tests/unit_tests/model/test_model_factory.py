import unittest
from unittest.mock import MagicMock

from src.models.model_factory import create_model
import src.models.model_factory as model_factory


class TestModelFactory(unittest.TestCase):
    def test_existing_model(self):
        # given
        mock_model = MagicMock()
        model_factory.MODEL_CREATORS["TestModelName"] = lambda: mock_model

        model_config = {"name": "TestModelName"}

        # when & then
        self.assertEqual(create_model(**model_config), mock_model)

    def test_non_existing_model(self):
        #  given
        model_config = {"name": "NonExisitingModel"}

        # when & then
        self.assertRaises(KeyError, lambda: model_factory.create_model(**model_config))
