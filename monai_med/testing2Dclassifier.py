import torch
from torch.utils.data import DataLoader, random_split
from monai.apps import MedNISTDataset
from medNIST import transform_image
from model import SimpleCNN
from model_utils import load_model, evaluate_model

testing_data = MedNISTDataset(root_dir="./data/MedNIST", section="test", transform=transform_image())
total_size_test = len(testing_data)
test_size = int(0.1 * total_size_test)
test_data, _ = random_split(testing_data, [test_size, total_size_test - test_size])
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)

model = SimpleCNN(num_classes=6)
model = load_model(model, "simple_cnn.pth")

evaluate_model(model, test_loader, class_names=["AbdomenCT", "BreastMRI", "ChestCT", "CXR", "HeadCT", "Hand"])  
