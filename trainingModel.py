import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd
from torch.utils.data import random_split

class TrafficSignModel(nn.Module): #defining custom model
    def __init__(self, num_classes):
        super(TrafficSignModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

labels_df = pd.read_csv("labels.csv") #Loading label CSV file
labels = dict(zip(labels_df["ClassID"], labels_df["Name"]))

transform = transforms.Compose([ #processing images
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

data_dir = "DATA" #Loading data
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

total_size = len(dataset) #splitting dataset to have train and testing
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instantiate the model
num_classes = len(labels)
model = TrafficSignModel(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / train_size
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

model.eval() #Evaluating
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Validation Accuracy: {100 * correct / total:.2f}%")

test_transform = transforms.Compose([ #Testing
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    image = Image.open(image_path)
    image = test_transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        model.eval()
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        sign_name = labels[predicted.item()]
    return sign_name

test_images = ["TEST/000_1_0001_1_j.png", "TEST/000_1_0005_1_j.png"]
for img_path in test_images:
    sign_name = predict_image(img_path)
    print(f"Image: {img_path} - Predicted sign: {sign_name}")

torch.save(model.state_dict(), "custom_traffic_sign_model.pth") #Saving model

model = TrafficSignModel(num_classes) #Loading model
model.load_state_dict(torch.load("custom_traffic_sign_model.pth"))
model.to(device)