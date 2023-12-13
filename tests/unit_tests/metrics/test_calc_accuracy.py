import torch
import unittest

from src.metrics.metrics import calc_accuracy


class TestCalcAccuracy(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2

    def test_max_score(self):
        # given
        mock_predictions = torch.tensor(
            [[0.1, 0.1, 0.8] for _ in range(self.batch_size)]
        )
        mock_labels = torch.tensor([2 for _ in range(self.batch_size)])

        # when
        accuracy = calc_accuracy(mock_predictions, mock_labels)

        # then
        self.assertEqual(accuracy.item(), 1.0)

    def test_min_score(self):
        # given
        mock_predictions = torch.tensor(
            [[0.1, 0.1, 0.8] for _ in range(self.batch_size)]
        )
        mock_labels = torch.tensor([1 for _ in range(self.batch_size)])

        # when
        accuracy = calc_accuracy(mock_predictions, mock_labels)

        # then
        self.assertEqual(accuracy.item(), 0.0)

    def test_case_score(self):
        # given
        mock_predictions = torch.tensor([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
        mock_labels = torch.tensor([1 for _ in range(self.batch_size)])

        # when
        accuracy = calc_accuracy(mock_predictions, mock_labels)

        # then
        self.assertEqual(accuracy.item(), 0.5)
