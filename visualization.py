import matplotlib.pyplot as plt
from medNIST import image_files_list, num_total, class_names, image_class


def plot_images():
    try:

        fig, axes = plt.subplots(3, 3, figsize=(8, 8)) # creating a 3x3 grid of subplots

        for i, k in enumerate(np.random.randint(0, num_total, size=9)):
            im = Image.open(image_files_list[k])
            arr = np.array(im)

            ax = axes[i // 3, i % 3]
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
            ax.set_title(class_names[image_class[k]])
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error loading/displaying images: {e}")


if __name__ =="__main__":
    plot_images()