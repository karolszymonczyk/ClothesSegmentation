import torch
from typing import Tuple


def calc_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    predicted_lables = torch.argmax(predictions, dim=1)
    accuracy_sum = torch.sum(predicted_lables == labels)
    batch_size = predictions.shape[0]
    return accuracy_sum / batch_size


def calc_iou(
    predictions: torch.Tensor, labels: torch.Tensor, threshold: float
) -> torch.Tensor:
    intersection, union = calc_intersection_and_union_per_sample(
        predictions, labels, threshold
    )
    iou = intersection / union
    return iou.mean()


def calc_dice_coeff(
    predictions: torch.Tensor, labels: torch.Tensor, threshold: float
) -> torch.Tensor:
    intersection, union = calc_intersection_and_union_per_sample(
        predictions, labels, threshold
    )
    dice_coefficient = (2.0 * intersection) / (union + intersection)
    return dice_coefficient.mean()


def calc_intersection_and_union_per_sample(
    mask1: torch.Tensor, mask2: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    binary_mask1 = mask1 >= threshold
    binary_mask2 = mask2 >= threshold

    intersection = torch.logical_and(binary_mask1, binary_mask2)
    union = torch.logical_or(binary_mask1, binary_mask2)

    batch_size = intersection.shape[0]
    intersection_per_sample = intersection.view(batch_size, -1)
    union_per_sample = union.view(batch_size, -1)

    intersection_sum = torch.sum(intersection_per_sample, dim=1)
    union_sum = torch.sum(union_per_sample, dim=1)

    return intersection_sum, union_sum
