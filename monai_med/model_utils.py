import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


