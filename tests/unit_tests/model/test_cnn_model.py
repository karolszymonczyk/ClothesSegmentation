import torch
import unittest

from src.models.cnn import CNN


class TestCNNModel(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.image_channels = 3
        self.image_heigth = 16
        self.image_width = 16

        self.mask_channels = 1

    def test_prediciton_shape(self):
        # given
        mock_data = torch.ones(
            self.batch_size, self.image_channels, self.image_heigth, self.image_width
        )
        cnn_model = CNN()

        # when
        prediction = cnn_model(mock_data)

        # then
        target_shape = torch.Size(
            [self.batch_size, self.mask_channels, self.image_heigth, self.image_width]
        )
        self.assertEqual(prediction.shape, target_shape)
