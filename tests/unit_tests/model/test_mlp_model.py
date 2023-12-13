import torch
import unittest

from src.models.mlp import MLP


class TestMLPModel(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2

        self.image_channels = 3
        self.image_heigth = 5
        self.image_width = 5

        self.input_dim = self.image_channels * self.image_heigth * self.image_width
        self.hidden_dim = 3
        self.output_dim = 3

    def test_prediciton_shape(self):
        # given
        mock_data = torch.ones(
            self.batch_size, self.image_channels, self.image_heigth, self.image_width
        )
        mlp_model = MLP(self.input_dim, self.hidden_dim, self.output_dim)

        # when
        prediction = mlp_model(mock_data)

        # then
        self.assertEqual(
            prediction.shape, torch.Size([self.batch_size, self.output_dim])
        )
