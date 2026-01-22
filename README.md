# AI Deepfake & Manipulated Content Verification Tool

## Overview

The rapid advancement of generative AI has led to a significant rise in deepfake and manipulated media, including images, videos, and audio. These are increasingly used in fraud, impersonation, misinformation, and social engineering attacks. Elderly and digitally less-aware users are especially vulnerable to such threats.

This project presents a **real, end-to-end AI-powered system** that verifies the authenticity of images and videos using **pretrained deepfake detection models**, explainable AI techniques, and asynchronous processing for heavy workloads.

The system is designed as a **deployable software product**, not a simulation.

---

## Key Features

* Image deepfake detection using pretrained models
* Video deepfake detection with temporal aggregation
* Confidence score with human-readable verdicts:

  * Real
  * Suspicious
  * Likely Fake
* Asynchronous video analysis using Redis + Celery
* Explainable AI support for images
* Privacy-first processing (no media permanently stored)
* API-driven architecture ready for web or mobile frontends

---

## Supported Media Types

| Media Type        | Status    |
| ----------------- | --------- |
| Images (JPG, PNG) | Supported |
| Videos (MP4)      | Supported |
| Audio             | Planned   |
| URLs / Links      | Planned   |

---

## Model Architecture

### Image Detection

* **Model:** XceptionNet (pretrained on FaceForensics++)
* **Input:** RGB image
* **Output:** Manipulation probability
* **Reason:** Widely used research baseline with strong performance on compressed media

### Video Detection

* Frame sampling from video
* Image-level inference on selected frames
* **Top-K temporal aggregation** to focus on highly manipulated frames
* Final video-level confidence score

This approach follows standard practices used in deepfake detection research and industry systems.

---

## Technology Stack

### Backend

* Python
* FastAPI
* PyTorch
* OpenCV
* Redis
* Celery

### Machine Learning

* XceptionNet (pretrained)
* Torchvision
* TIMM
* NumPy
* Scikit-learn (evaluation and metrics)

### Infrastructure

* Asynchronous task queue (Celery)
* Message broker (Redis)
* CPU-based inference (GPU optional)

---

## Project Structure

```
media-detector/
├── backend/
│   ├── api/                # FastAPI routes
│   ├── ml/                 # ML models and evaluation
│   ├── tasks/              # Celery background tasks
│   ├── utils/              # Helpers and utilities
│   ├── main.py             # FastAPI entry point
│   └── celery_app.py       # Celery configuration
├── pretrained/             # Pretrained model weights
│   └── xception_c23.p
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AI-Deepfake-Manipulated-Content-Verification-Tool/media-detector
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages include:

* torch
* torchvision
* timm
* fastapi
* uvicorn
* celery
* redis
* opencv-python
* pillow
* scikit-learn

---

### 3. Download Pretrained Model Weights

Download the FaceForensics++ pretrained XceptionNet weights:

```
https://github.com/ondyari/FaceForensics/releases/download/v1.0/xception_c23.p
```

Place the file here:

```
media-detector/pretrained/xception_c23.p
```

---

### 4. Start Redis

Using Docker:

```bash
docker run -d -p 6379:6379 redis
```

---

### 5. Start Celery Worker (Windows)

```bash
celery -A backend.celery_app.celery_app worker --loglevel=info --pool=solo
```

---

### 6. Start FastAPI Server

```bash
uvicorn backend.main:app
```

API documentation will be available at:

```
http://127.0.0.1:8000/docs
```

---

## Evaluation

The system is evaluated using the **FaceForensics++ dataset (C23 compression)**.

Metrics reported:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Evaluation is performed using **real inference**, not simulated outputs.

---

## Privacy and Security

* Media files are processed in memory or temporary storage only
* No user content is permanently stored
* No personal data collection
* Designed to be compliant with privacy-first principles

---

## Limitations

* Audio and live call analysis are not yet implemented
* Video detection relies on image-based models with temporal aggregation
* Performance depends on hardware (CPU vs GPU)

