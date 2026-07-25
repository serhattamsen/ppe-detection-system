# PPE Detection System

**AI-powered Personal Protective Equipment (PPE) detection for workplace safety monitoring, built with YOLO11.**

🔗 **[Live Demo](https://serhat-ppe-detection.streamlit.app)** — try it live by uploading your own workplace image.

---

An AI-powered computer vision system that monitors workers' Personal Protective Equipment (PPE) compliance in real time through a web interface. Upload an image and the system detects safety equipment (or its absence) and flags violations automatically.

<!-- Buraya kendi çektiğin veya telifsiz bir görselle yaptığın tespitin ekran görüntüsünü ekle -->
<!-- Örnek: ![Demo](docs/demo.png) -->

## Features

- YOLO11-based real-time object detection
- Image, video, and live camera support (webcam / IP camera / RTSP)
- Detection of hardhat, safety vest, mask, gloves, and goggles, with automatic violation alerts
- Fall detection
- Ensemble mode: hardhat detection is handled by a dedicated high-accuracy model, while other equipment is detected by a multi-class model
- Automatic image capture and alarm history when a violation occurs
- Clean web interface built with Streamlit

## Models

| Model            | Classes      | mAP50        | Description                    |
| ---------------- | ------------ | ------------ | ------------------------------ |
| `baret_model.pt` | head, helmet | 0.92 (head)  | Hardhat only, high accuracy    |
| `kkd_model.pt`   | 14 classes   | 0.73         | YOLO11n, multi-equipment       |
| `kkd_model_s.pt` | 14 classes   | 0.74         | YOLO11s, multi-equipment       |

## Tech Stack

Python · Ultralytics YOLO11 · OpenCV · Streamlit · PyTorch

## How It Works

The Streamlit web app runs one or more trained YOLO11 models on the uploaded image. In ensemble mode, a dedicated high-accuracy model detects hardhats while a multi-class model detects the remaining equipment. Detected classes starting with `NO-` (e.g. `NO-Safety Vest`) or matching fall/head states are treated as safety violations and reported with a live compliance summary.

## Recommended Settings

For the best results in the live demo, use the following settings in the sidebar:

- **Mode (Calisma kipi):** `Birlesik (baret + KKD)` — ensemble mode combines the
  dedicated high-accuracy hardhat model with the multi-class equipment model.
- **Confidence threshold (Guven esigi):** `0.35` — a good balance between catching
  real violations and avoiding false positives.
- **KKD model:** `kkd_model_s.pt` — the YOLO11s model, which gives the best overall accuracy.

In this configuration, hardhat detection is handled by the dedicated hardhat model,
while all other equipment (vest, mask, gloves, goggles) is detected by the KKD model.

## Getting Started

```bash
# Install dependencies
pip install ultralytics streamlit opencv-python

# Run the app
streamlit run app.py
```

Then open the local URL shown in the terminal and use the **"Goruntu analizi"** (Image analysis) tab to upload an image.

> **Note:** The live camera tab works only when running locally, since a hosted server has no camera hardware. In the online demo, please use the image-upload tab.

## Dataset

Trained on a subset of ~10,700 images from **Roboflow Universe – PPE Combined Model** (CC BY 4.0).

## Results

The multi-class model performs best on large objects such as hardhats and safety vests, while smaller items (goggles, gloves) show lower accuracy due to class imbalance in the dataset. The dedicated hardhat model reaches high accuracy on head/helmet detection.

## About

Developed as a computer vision project focused on real-time workplace safety monitoring.
<!-- İstersen kendi katkını ekle, ör: Model training, ensemble logic, and the Streamlit interface were built by me. -->
