import os
from monai.apps import MedNISTDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ResizeD, ToTensord, RandFlipd
from PIL import Image
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
        print("Download error:", e)
        return None


   

