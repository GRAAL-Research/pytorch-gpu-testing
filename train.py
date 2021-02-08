# Import the Poutyne Model
from poutyne import Model
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Instanciate the MNIST dataset
train_valid_dataset = MNIST('.', train=True, download=True, transform=ToTensor())
test_dataset = MNIST('.', train=False, download=True, transform=ToTensor())
train_dataset, valid_dataset = random_split(train_valid_dataset, [50_000, 10_000],
                                            generator=torch.Generator().manual_seed(42))
# Select CUDA device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Running on {torch.cuda.get_device_name(device)}")
else:
    print("Running on CPU")

# Define the network
network = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

# Train
model = Model(network, 'sgd', 'cross_entropy',
              batch_metrics=['accuracy'],
              epoch_metrics=['f1'],
              device=device)
# Change the number of epochs to find the optimum value for your work
model.fit_dataset(
    train_dataset,
    valid_dataset,
    epochs=1,
    batch_size=32,
    num_workers=2
)

model.evaluate_dataset(test_dataset)
