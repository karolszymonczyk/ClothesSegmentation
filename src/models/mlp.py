import torch


class MLP:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 1e-3) -> None:
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
    
    def train(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        preds = self.forward(X)
        loss = self.criterion(preds, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
