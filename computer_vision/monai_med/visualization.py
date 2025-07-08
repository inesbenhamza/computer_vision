import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_dataset_info():
    root_dir = "./data/MedNIST"
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

    return image_files_list, image_class, class_names, num_each

def plot_random_images(image_files_list, image_class, class_names):
    num_total = len(image_files_list)
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    for i, k in enumerate(random.sample(range(num_total), 9)):
        im = Image.open(image_files_list[k])
        arr = np.array(im)

        ax = axes[i // 3, i % 3]
        ax.imshow(arr, cmap="gray")
        ax.set_title(class_names[image_class[k]])
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_files_list, image_class, class_names, num_each = load_dataset_info()

    image_width, image_height = Image.open(image_files_list[0]).size
    print(f"Total image count: {len(image_class)}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")

    plot_random_images(image_files_list, image_class, class_names)