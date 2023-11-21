import torch
import torch.nn.functional as F

from src.models.mlp import MLP
from src.utils import batch_data

# def fizz_buzz(num):
#     if num % 15 == 0: return "FizzBuzz"
#     if num % 3 == 0: return "Fizz"
#     if num % 5 == 0: return "Buzz"
#     return ""

def gen_fizzbuzz_labels(size: int) -> torch.Tensor:
    labels_map = {0: "", 1: "fizz", 2: "buzz", 3: "fizz & buzz"}
    labels = []
    for i in range(size):
        if i <= 0:
            labels.append(0)
            continue
        res = 0
        if i % 3 == 0:
            res += 1
        if i % 5 == 0:
            res += 2
        labels.append(res)
    return torch.tensor(labels), labels_map

def get_fizzbuzz_data(size: int) -> torch.Tensor:
    x = torch.arange(0, size).unsqueeze(1)
    y, y_map = gen_fizzbuzz_labels(size)

    num_classes=len(y_map.keys())
    y_encoded = F.one_hot(y, num_classes=num_classes).float()
    return x, y_encoded, y_map

def train() -> None:
    x, y, y_map = get_fizzbuzz_data(100)
    num_classes = len(y_map.keys())
    lr = 1e-3
    batch_size = 100
    epochs = 10_000
    model = MLP(input_size=1, hidden_size=256, output_size=num_classes, lr=lr)

    history = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in zip(batch_data(x, batch_size), batch_data(y, batch_size)):
            loss = model.train(batch_x.float(), batch_y.float())
            epoch_loss += loss
        history.append(epoch_loss)
        if epoch % 1000:
            print(epoch_loss)


if __name__ == "__main__":
    train()
