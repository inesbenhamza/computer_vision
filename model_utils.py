import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision.utils import make_grid

def save_model(model, path="simple_cnn.pth"):
    torch.save(model.state_dict(), path)
    #print(f"Model saved to {path}")

def load_model(model, path="simple_cnn.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"]
            labels = batch["label"]
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n Accuracy: {acc:.4f}")
    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("\n Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

def show_batch(images, labels, class_names):
    img_grid = make_grid(images, nrow=8, normalize=True)
    np_img = img_grid.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(12, 4))
    plt.imshow(np_img)
    plt.title(" | ".join([class_names[l] for l in labels]))
    plt.axis('off')
    plt.show()
