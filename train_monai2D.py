import torch
import torch.nn as nn
import torch.optim as optim
from monai.data import DataLoader
from monai.apps import MedNISTDataset
from medNIST import download_dataset, transform_image
from torch.utils.data import random_split
from models import SimpleCNN
from model_utils import save_model


training_data = download_dataset(root_dir="./data/MedNIST")
total_size = len(training_data)
train_size = int(0.1 * total_size)
train_data, _ = random_split(training_data, [train_size, total_size - train_size])
data_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)


model = SimpleCNN(num_classes=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for batch in data_loader:
        images = batch["image"]
        labels = batch["label"]
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

save_model(model, "simple_cnn.pth")
