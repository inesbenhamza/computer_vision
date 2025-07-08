import cv2
import torch
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

import torchvision.transforms as T
import urllib.request

image_path = "/Users/inesbenhamza/Desktop/computer vision /Opencv_test/IMG_4335.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR) #load image in bgr format 

#transform image to tensor
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

print (image.shape)
#(4032, 3024, 3) so its an image of 4032 pixels in height, 3024 pixels in width, and 3 color channels (RGB).
image_tensor= torch.from_numpy(image) #this converts the numpy array to a PyTorch tensor
image_tensor = image_tensor.permute(2, 0, 1)  # Change shape from HWC to CHW

#Permuting reorders the tensor's dimensions from H×W×C (Height, Width, Channels) into C×H×W (Channels, Height, Width), which is what PyTorch’s conv layers expect.

image_tensor = image_tensor.float()  # Convert to float tensor
print(image_tensor.shape)
image_tensor = image_tensor.div(255.0) #normalize the tensor values (pixel intensity values) to the range [0, 1]

normalize = T.Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize (image_tensor)  # Normalize the tensor using ImageNet statistics
#output[c] = (input[c] - mean[c]) / std[c]
#making your input distribution zero-centered with unit variance per channel, which helps pretrained models converge.
tensor = image_tensor.unsqueeze(0)  # Add batch dimension, required because PyTorch models expect inputs of shape [batch_size, channels, height, width].v
print (tensor.shape) #([1, 3, 4032, 3024])


device = torch.device("mps " if torch.cuda.is_available() else "cpu")
tensor = tensor.to(device)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  

model = model.to(device)  

def new_func():
    return torch.nn.functional

with torch.no_grad(): 
    out  = model(tensor)
    print (out.shape)  #[1, 1000] so one column and 1000 rows, where each row corresponds to a class in the ImageNet dataset.
    print(out[0].shape)
    probabilities = new_func().softmax(out[0], dim=0)  # Apply softmax to get probabilities


top5_prob, top5_catid = probabilities.topk(5)

labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = []
with urllib.request.urlopen(labels_url) as f:
    labels = [line.decode().strip() for line in f.readlines()]

print("Top-5 predictions:")
for prob, cid in zip(top5_prob, top5_catid):
    print(f"{labels[cid]}: {prob.item():.4f}")
