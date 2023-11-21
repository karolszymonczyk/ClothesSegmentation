import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.mlp import MLP



def train() -> None:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Set input size based on the Fashion MNIST images (28x28)
    input_size = 28 * 28
    hidden_size = 128
    num_classes = 10  # Number of classes in Fashion MNIST
    lr = 1e-3

    model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=num_classes, lr=lr)

    epochs = 5
    for epoch in range(epochs):
        for images, labels in train_loader:
            loss = model.train(images, labels)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}')

if __name__ == "__main__":
    train()
