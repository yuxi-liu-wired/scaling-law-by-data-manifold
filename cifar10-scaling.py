import torch
import torch.nn as nn
from tqdm import trange
from torchmetrics.functional import accuracy
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Check if GPU is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## --------------------------------------------------------------------------------
## Define the model
## --------------------------------------------------------------------------------

class ConvNet(nn.Module):
    def __init__(self, n):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, n, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(n, n*2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(n*2, n*2, kernel_size=3, padding=0)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n*2 * 6 * 6, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x

def model_parameter_count(m):
  return sum(p.numel() for p in m.parameters())

## --------------------------------------------------------------------------------
## Define the dataset
## --------------------------------------------------------------------------------

# Data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

## --------------------------------------------------------------------------------
## Define training run
## --------------------------------------------------------------------------------

from torchmetrics.classification import Accuracy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def training_loop(channel_count):
    model = ConvNet(channel_count)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model.to(device)

    num_epochs = 50

    # Training loop
    progress_bar = trange(num_epochs)

    metric = Accuracy(task="multiclass", num_classes=10).to(device)
    metric.reset()

    experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f'logs/n_{channel_count}/{experiment_name}')

    metadata = {
        'Channel count': channel_count,
        'Model size': model_parameter_count(model),
        'Epochs': num_epochs,
    }

    for key, value in metadata.items():
        writer.add_text(key, str(value))

    for epoch in progress_bar:
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            metric(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        writer.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/Train', metric.compute().item(), epoch)
        metric.reset()

        # Calculate validation loss at the end of each epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                val_loss += criterion(outputs, labels).item()
                metric(outputs, labels)

        writer.add_scalar('Loss/Validation', val_loss / len(val_loader), epoch)
        writer.add_scalar('Accuracy/Validation', metric.compute().item(), epoch)
        metric.reset()

    progress_bar.close()
    writer.close()

if __name__ == "__main__":
    for i in range (1,21):
        training_loop(i)