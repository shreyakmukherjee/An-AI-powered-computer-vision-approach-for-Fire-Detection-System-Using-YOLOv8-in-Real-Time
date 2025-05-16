
---

## ✅ Features

- Real-time fire detection via webcam
- Custom-trained YOLOv8 model
- Fast and lightweight inference
- Clear yellow-colored fire annotation
- Simple and effective fire localization

---

## 📦 Dataset

The dataset used in this project contains images and video frames of fire, annotated with **yellow bounding boxes**.

📥 **Download the dataset here:**  
[🔗 Kaggle Fire Dataset](https://universe.roboflow.com/-jwzpw/continuous_fire)  

---

## 🧠 Approach

1. **Data Preparation**:
    - Collected and annotated fire images using yellow boxes.
    - Dataset structured in YOLO format.

2. **Model Training**:
    - Utilized [YOLOv8](https://github.com/ultralytics/ultralytics).
    - Trained using the notebook `fire_detection.ipynb`.
    - The best-performing model saved as `best.pt`.

3. **Real-Time Inference**:
    - Open Vs Code and create `main.py` (Code is already given in this Repo.)
    - The script `main.py` loads the model and uses a webcam feed (`source=0`) to detect fire in real time.

---

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/fire-detection-yolov8.git
cd fire-detection-yolov8
```
2. **pip install ultralytics opencv-python**
```bash
pip install ultralytics opencv-python
```
## 🎯 How to Run
Make sure your webcam is working.
```bash
python main.py
```
A window will open showing your webcam feed with fire detection overlay (if fire is present).

## 🔧 main.py - Real-Time Detection Script
Make sure your webcam is working.
```bash
from ultralytics import YOLO

model = YOLO('best.pt')
model.predict(source=0, imgsz=640, conf=0.6, show=True)
```

## 📒 fire_detection.ipynb

This notebook includes:
Dataset loading
YOLOv8 training pipeline
Model evaluation
Visualization of detection results

# 1️⃣ Install dependencies (if not already installed)
```bash
!pip install ultralytics opencv-python
```
# 2️⃣ Import necessary modules
```bash
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
```
# 3️⃣ Check Ultralytics version
```bash
import ultralytics
ultralytics.checks()
```
Here you can find th Requirement.txt file
