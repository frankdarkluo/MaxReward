# Import necessary libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
import torch.multiprocessing as mp


# Function to set up the distributed environment
def setup(rank, world_size):
    # Set the IP address and port of the master node
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the distributed process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


# Function to clean up the distributed environment
def cleanup():
    dist.destroy_process_group()


# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Define the forward pass of the neural network
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(rank, world_size):
    # Set up the distributed environment
    setup(rank, world_size)

    # Create dataset and dataloader
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Create sampler for distributed training
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    # Create dataloader with distributed sampler
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, sampler=sampler, num_workers=2)

    # Create the model, optimizer, and loss function
    device = torch.device(f"cuda:{rank}")
    model = SimpleNet().to(device)

    # Wrap model with DistributedDataParallel
    ddp_model = DDP(model, device_ids=[rank])

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimization step
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Update the model parameters
            optimizer.step()

            # Update the running loss
            running_loss += loss.item()

            # Print the loss every 2000 mini-batches
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                running_loss = 0.0

    # Clean up the distributed environment
    cleanup()

def main():
    # Get the number of GPUs available
    world_size = torch.cuda.device_count()
    # Spawn one process per GPU and start the training
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()