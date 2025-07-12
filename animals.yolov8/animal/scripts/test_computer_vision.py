import os, random, shutil
from ultralytics import YOLO
import torch
import os
import yaml


base_dir = "/Users/inesbenhamza/Desktop/computer vision /Opencv_test/animals.v1i.yolov8"

paths = {split: {
    "images": os.path.join(base_dir, split, "images"),
    "labels": os.path.join(base_dir, split, "labels"),
} for split in ("train","valid","test")}
 

train_images, train_labels = paths["train"].values()
valid_images, valid_labels = paths["valid"].values()
test_images,  test_labels  = paths["test"].values()


print(paths["train"]["images"])  # verify this folder exists

nc = 10
names = ['cat','chicken','cow','dog','fox','goat','horse','person','racoon','skunk']




def sample_subset(img_dir, lbl_dir, out_dir, ratio=0.1):
    os.makedirs(out_dir, exist_ok=True)
    imgs = sorted(os.listdir(img_dir))
    k = max(1, int(len(imgs)*ratio))
    subset = random.sample(imgs, k)
    for fn in subset:
        shutil.copy(os.path.join(img_dir, fn), os.path.join(out_dir, fn))
        lbl = fn.rsplit(".",1)[0] + ".txt"
        shutil.copy(os.path.join(lbl_dir, lbl), os.path.join(out_dir, lbl))
    return out_dir

train_subset = sample_subset(train_images, train_labels,
                             "/tmp/train_subset", ratio=0.1)


data = dict(
    train=train_subset,
    val=valid_images,
    test=test_images,
    nc=nc,
    names=names
)


device = "mps" if torch.backends.mps.is_available() else "cpu"


with open("tmp_data.yaml", "w") as f:
    yaml.dump(data, f)


model = YOLO("yolov8n.pt")
model.train(
            data="tmp_data.yaml",
            epochs=5,
            imgsz=640,
            batch=8,
            project="runs/train",
            name="animals_subset", 
            device=device
            )

results = model.val(data="tmp_data.yaml", imgsz=640, device=device)

print(f"Results: {results}")
model.export(format="onnx", dynamic=True, simplify=True)
model.save("animals_subset.pt")





