# ppe-detection-system
AI-powered PPE (Personal Protective Equipment) detection system - Real-time workplace safety monitoring with YOLO11
# PPE Detection System

An AI-powered computer vision system that monitors workers' Personal 
Protective Equipment (PPE) compliance in real time through a web interface.

## Features
- YOLO11-based object detection
- Image, video, and live camera support (webcam / IP camera / RTSP)
- Detection of hardhat, safety vest, mask, gloves, goggles with violation alerts
- Fall detection
- Automatic image capture and alarm history on violations
- Web interface built with Streamlit

## Models
| Model | Classes | mAP50 | Description |
|-------|---------|-------|-------------|
| baret_model.pt | head, helmet | 0.92 (head) | Hardhat only, high accuracy |
| kkd_model.pt | 14 classes | 0.73 | YOLO11n, multi-equipment |
| kkd_model_s.pt | 14 classes | 0.74 | YOLO11s, multi-equipment |

The system supports an ensemble mode where hardhat detection is handled 
by the dedicated high-accuracy model, while other equipment is detected 
by the multi-class model.

## Tech Stack
Python, Ultralytics YOLO11, OpenCV, Streamlit

## Installation

pip install ultralytics streamlit opencv-python
streamlit run app.py

## Dataset
Roboflow Universe - PPE Combined Model (CC BY 4.0)

## Results
Trained on a subset of ~10,700 images. The multi-class model performs 
best on large objects (hardhat, vest), while smaller items (goggles, 
gloves) show lower accuracy due to class imbalance in the dataset.
