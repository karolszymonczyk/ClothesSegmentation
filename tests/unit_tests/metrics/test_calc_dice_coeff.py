import torch
import unittest

from src.metrics.metrics import calc_dice_coeff


class TestCalcDiceCoeff(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.mask_heigth = 3
        self.mask_width = 3
        self.threshold = 0.8

    def test_max_score(self):
        # given
        mock_predictions = torch.ones(
            self.batch_size, self.mask_heigth, self.mask_width
        )
        mock_labels = mock_predictions.clone()

        # when
        dice_coeff = calc_dice_coeff(
            mock_predictions, mock_labels, threshold=self.threshold
        )

        # then
        self.assertEqual(dice_coeff.item(), 1.0)

    def test_max_score_with_float(self):
        # given
        mock_predictions = torch.empty(
            self.batch_size, self.mask_heigth, self.mask_width
        )
        mock_predictions.fill_(0.85)
        mock_labels = torch.ones(self.batch_size, self.mask_heigth, self.mask_width)

        # when
        dice_coeff = calc_dice_coeff(
            mock_predictions, mock_labels, threshold=self.threshold
        )

        # then
        self.assertEqual(dice_coeff.item(), 1.0)

    def test_min_score(self):
        # given
        mock_predictions = torch.ones(
            self.batch_size, self.mask_heigth, self.mask_width
        )
        mock_labels = torch.zeros(self.batch_size, self.mask_heigth, self.mask_width)

        # when
        dice_coeff = calc_dice_coeff(
            mock_predictions, mock_labels, threshold=self.threshold
        )

        # then
        self.assertEqual(dice_coeff.item(), 0.0)

    def test_min_score_from_float(self):
        # given
        mock_predictions = torch.empty(
            self.batch_size, self.mask_heigth, self.mask_width
        )
        mock_predictions.fill_(0.65)
        mock_labels = torch.ones(self.batch_size, self.mask_heigth, self.mask_width)

        # when
        dice_coeff = calc_dice_coeff(
            mock_predictions, mock_labels, threshold=self.threshold
        )

        # then
        self.assertEqual(dice_coeff.item(), 0.0)

    def test_case_score(self):
        # given
        mock_predictions = torch.stack(
            [
                torch.zeros(self.mask_heigth, self.mask_width),
                torch.ones(self.mask_heigth, self.mask_width),
            ]
        )
        mock_labels = torch.ones(self.batch_size, self.mask_heigth, self.mask_width)

        # when
        dice_coeff = calc_dice_coeff(
            mock_predictions, mock_labels, threshold=self.threshold
        )

        # then
        self.assertEqual(dice_coeff.item(), 0.5)
