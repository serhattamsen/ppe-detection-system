# PPE Detection System

AI-powered Personal Protective Equipment (PPE) detection for workplace safety 
monitoring, built with YOLO11.

🔗 **[Live Demo](https://serhat-ppe-detection.streamlit.app)** — upload a workplace image and see it in action.

A computer vision system that monitors workers' PPE compliance through a web 
interface. It detects safety equipment (or its absence), raises violation 
alerts, logs incidents to a database, and visualizes safety analytics per user.

## Features

- **YOLO11-based object detection** — 14 equipment classes
- **Multiple input sources** — image upload, webcam, IP camera (RTSP), video files
- **Multi-camera support** — define and switch between named camera sources
- **User authentication** — each user sees only their own detection history
- **Analytics dashboard** — most frequently missing PPE, violations by camera, hourly distribution
- **Ensemble mode** — hardhat detection handled by a dedicated high-accuracy model, other equipment by the multi-class model
- **Incident logging** — automatic image capture and database record on violation
- **Configurable alerting** — adjustable confidence threshold, frame skipping, and alarm cooldown

## Detected Classes

| Compliant | Violation |
|-----------|-----------|
| Hardhat | NO-Hardhat |
| Safety Vest | NO-Safety Vest |
| Mask | NO-Mask |
| Gloves | NO-Gloves |
| Goggles | NO-Goggles |
| — | Fall-Detected |

Context objects: Person, Ladder, Safety Cone

## Models

| Model | Architecture | Classes | mAP50 | Notes |
|-------|--------------|---------|-------|-------|
| `baret_model.pt` | YOLO11n | 3 | 0.92 (head) | Hardhat only, high accuracy |
| `kkd_model.pt` | YOLO11n | 14 | 0.734 | Multi-equipment, lightweight |
| `kkd_model_s.pt` | YOLO11s | 14 | 0.741 | Multi-equipment, larger backbone |

## Results

Trained on a randomly sampled subset of ~10,700 images (35% of the full dataset) 
for 25–60 epochs at 640×640 input resolution.

**Per-class performance (mAP50-95, YOLO11s):**

| Class | Score | Class | Score |
|-------|-------|-------|-------|
| Ladder | 0.777 | Hardhat | 0.486 |
| Person | 0.733 | Fall-Detected | 0.477 |
| Goggles | 0.553 | Gloves | 0.461 |
| NO-Goggles | 0.502 | NO-Hardhat | 0.446 |
| Safety Vest | 0.432 | NO-Mask | 0.367 |
| NO-Gloves | 0.366 | Mask | 0.335 |
| Safety Cone | 0.324 | NO-Safety Vest | 0.087 |

**Key findings:**

- Scaling from YOLO11n to YOLO11s produced only a marginal gain (0.734 → 0.741), 
  indicating that the bottleneck is dataset class imbalance rather than model capacity.
- Classes with fewer training instances (`NO-Safety Vest`: 505 samples) perform 
  significantly worse than well-represented ones (`Hardhat`: 9,940 samples).
- An earlier training run using Ultralytics' `fraction` parameter yielded mAP50 of 
  only 0.101, because the parameter selects the first N% of a sorted file list — and 
  the dataset is ordered by source, leaving 7 classes entirely absent from training. 
  Replacing it with random sampling raised mAP50 to 0.734.
- The system performs best on real-world site imagery; accuracy drops on studio-style 
  photographs with plain backgrounds, reflecting the training data distribution.

## Recommended Settings

For best results when testing the demo:

| Setting | Recommended | Notes |
|---------|-------------|-------|
| Mode | Ensemble (hardhat + PPE) | Combines the high-accuracy hardhat model with the multi-class model |
| Confidence threshold | 0.30 – 0.40 | Lower values catch more objects but increase false positives |
| Model (single mode) | `kkd_model_s.pt` | Larger backbone, slightly better on small objects |

**Image selection matters.** The models were trained on real construction site 
and industrial imagery. Photographs with plain or studio backgrounds produce 
noticeably weaker results, as they fall outside the training distribution. 
For a representative test, use images of workers on an actual site.

For live camera use:

| Setting | Recommended | Notes |
|---------|-------------|-------|
| Frame skip | 2 – 4 | Higher values keep the stream smooth on CPU-only machines |
| Alarm cooldown | 10 s | Prevents hundreds of duplicate captures for a single ongoing violation |

## Tech Stack

Python · Ultralytics YOLO11 · OpenCV · Streamlit · SQLite · bcrypt · pandas

## Installation

```bash
git clone https://github.com/serhattamsen/ppe-detection-system.git
cd ppe-detection-system
pip install -r requirements.txt
streamlit run app.py
```

Register an account on first launch, then log in.

### Configuring cameras

In the live camera tab, define sources one per line:

```
Entrance = 0
Warehouse = rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101
```

Use `0` for the default webcam. RTSP paths vary by manufacturer — check your 
camera's documentation.

## Notes

- The live demo supports image analysis and dashboard features only. Live camera 
  detection requires local installation, as cloud servers have no camera access.
- The database is ephemeral on Streamlit Cloud and resets when the app restarts.
- Detection is object-based rather than person-based: the system reports how many 
  violations are present, not which individual each belongs to.

## Dataset

[PPE Combined Model](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model) 
— Roboflow Universe (CC BY 4.0), 44,002 annotated images.

## Future Work

- Person-to-equipment association for per-worker compliance tracking
- Higher input resolution (960px) to improve small-object detection
- Additional equipment classes (ear protection, safety harness, footwear)
- Concurrent multi-camera processing
- Smoke and fire detection
