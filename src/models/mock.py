import torch
import torch.nn as nn


class Mock(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(nn.Flatten(), nn.Linear(input_dim, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
