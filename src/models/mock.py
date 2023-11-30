import torch
import torch.nn as nn


class Mock(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images
