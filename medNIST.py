from monai.apps import MedNISTDataset
import os
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ResizeD, ToTensord, RandFlipd
from PIL import Image 
import numpy as np 
from functools import lru_cache


def transform_image():
    return Compose([
        LoadImaged(keys=["image"]),             
        EnsureChannelFirstd(keys=["image"]),
        ResizeD(keys=["image"], spatial_size=(64, 64)),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"]),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    ])


@lru_cache(maxsize=None)
def download_dataset(root_dir): 
    try: 
        dataset = MedNISTDataset(root_dir=root_dir, section="training", download=True, transform=transform_image())
        print("Dataset downloaded successfully.")
        return dataset
    except Exception as e:
        print("Error has occurred:", e)
        return None


def main():
    root_dir = "./data/MedNIST"

    if os.path.exists(root_dir) and not os.path.isdir(root_dir):
        os.remove(root_dir)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)



    data_dir = os.path.join(root_dir, "MedNIST")

    class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
    num_class = len(class_names)

    image_files = [
        [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
        for i in range(num_class)
    ]

    num_each = [len(image_files[i]) for i in range(num_class)]

    image_files_list = []
    image_class = []

    for i in range(num_class):
        image_files_list.extend(image_files[i])
        image_class.extend([i] * num_each[i])

    image_width, image_height = Image.open(image_files_list[0]).size


    def print_info(image_files_list, image_class, class_names, num_each, image_width, image_height):
        print(f"Total image count: {len(image_class)}")
        print(f"Image dimensions: {image_width} x {image_height}")
        print(f"Label names: {class_names}")
        print(f"Label counts: {num_each}")

    print_info(image_files_list, image_class, class_names, num_each, image_width, image_height)

    

if __name__ == "__main__":
    main()




if __name__ == "__main__": 

