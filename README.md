# Real-Time Fire Detection using YOLOv8

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shreyakmukherjee/An-AI-powered-computer-vision-approach-for-Fire-Detection-System-Using-YOLOv8-in-Real-Time/blob/main/fire_detection.ipynb)


It is a real-time fire detection system built using the YOLOv8 object detection model and trained on a publicly available fire dataset from Kaggle. The project leverages the speed and accuracy of deep learning to detect fire or flames from live video streams, such as a webcam feed. By processing frames in real-time, the system can help in early flame detection, which is crucial for fire safety and surveillance applications.

The model is trained on a labeled Kaggle dataset, which includes various fire scenarios in both indoor and outdoor environments. After training, the best model weights are used to perform live inference via webcam, displaying bounding boxes around detected fire regions.

This solution is lightweight, efficient, and can be deployed on standard hardware, making it suitable for smart surveillance systems, forest fire monitoring, and real-time hazard alerting.

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
