import os
import torch
from torchvision import datasets, transforms

# Define a transformation pipeline for images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),         #to tensor
])

# Create a dataset and dataloader
data_dir = "DATA"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split the dataset into training and validation sets
from torch.utils.data import random_split

total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
