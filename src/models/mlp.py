import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 1e-3) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def train(self, images: torch.Tensor, lables: torch.Tensor) -> torch.Tensor:
        preds = self.forward(images)
        loss = self.criterion(preds, lables)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
