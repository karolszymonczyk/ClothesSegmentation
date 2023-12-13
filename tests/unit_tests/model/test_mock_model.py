import torch
import unittest

from src.models.mock import Mock


class TestMockModel(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.image_channels = 3
        self.image_heigth = 5
        self.image_width = 5

        self.input_dim = self.image_channels * self.image_heigth * self.image_width

    def test_prediciton_shape(self):
        # given
        mock_data = torch.ones(
            self.batch_size, self.image_channels, self.image_heigth, self.image_width
        )
        mock_model = Mock(self.input_dim)

        # when
        prediction = mock_model(mock_data)

        # then
        self.assertEqual(prediction.shape, torch.Size([self.batch_size, 1]))
