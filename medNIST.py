from monai.apps import MedNISTDataset
import os
from monai.data import DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ResizeD, ToTensord, RandFlipd
import matplotlib.pyplot as plt
from PIL import Image 
import numpy as np 



root_dir = "./data/MedNIST"

### fixing the directory using os.makedirs
if os.path.exists(root_dir) and not os.path.isdir(root_dir):
    os.remove(root_dir)  # in case it's a file

if not os.path.exists(root_dir):
    os.makedirs(root_dir, exist_ok=True) 


#downloading the data 
def download_dataset(): 
    try: 
        dataset = MedNISTDataset(root_dir="./data/MedNIST", section="training", download=True)
        print ("Dataset downloaded successfully.")
    except Exception as e :
        print("error has occured:", e)




def transorm_image():
    return Compose([
        LoadImaged(keys=["image"]),             
        EnsureChannelFirstd(keys=["image"]),
        ResizeD(keys=["image"], spatial_size=(64, 64)),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"]),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),  # Randomly flip along the first axis
    ])


train_ds = MedNISTDataset(
    root_dir=root_dir,
    section="training",
    transform = transorm_image())



data_dir = "./data/MedNIST/MedNIST/"

class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))) #list all the subdirectories in the data_dir, each subdirectory is a class
num_class = len(class_names) 
image_files = [
    [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
    for i in range(num_class)
] # for each class, list all the image files in that class directory, so its alist of list 
num_each = [len(image_files[i]) for i in range(num_class)] # get the number of images in each class
image_files_list = []
image_class = []
for i in range(num_class):
    image_files_list.extend(image_files[i])
    image_class.extend([i] * num_each[i]) #flattens the image_files list (which is a list of lists) and associates each image with its corresponding class label.
num_total = len(image_class) # in total in all classes ho wmany images do we have 
image_width, image_height = Image.open(image_files_list[0]).size # open the first image and get its dimensions



def print_info():
    print(f"Total image count: {num_total}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")



if __name__ == "__main__": 

