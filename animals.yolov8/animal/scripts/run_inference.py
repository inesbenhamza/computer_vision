from ultralytics import YOLO
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO("animals_subset.pt")

# Run inference on a single image
results = model.predict(
    source="/Users/inesbenhamza/Desktop/computer vision /Opencv_test/animals.v1i.yolov8/IMG_4335.jpeg",
    imgsz=640,
    conf=0.25,
    device=device,
    save=True,
    project="runs/predict_single",
    name="animals_subset_inference"
)


print(results)


boxes = results[0].boxes
names = results[0].names

if boxes is not None and boxes.cls is not None:
    for cls_id, conf, box in zip(boxes.cls, boxes.conf, boxes.xyxy):
        class_name = names[int(cls_id)]
        print(f"Detected {class_name} with confidence {conf:.2f} at {box.tolist()}")
else:
    print("No objects detected.")