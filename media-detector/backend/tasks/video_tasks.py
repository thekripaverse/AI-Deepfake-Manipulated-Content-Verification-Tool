from backend.celery_app import celery_app
import cv2
import numpy as np
from backend.ml.image_model import analyze_image
import tempfile
import os
from PIL import Image


def analyze_video_core(video_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    scores = []

    while cap.isOpened() and frame_count < 10:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ✅ OpenCV → PIL conversion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        result = analyze_image(frame_pil)
        scores.append(result["confidence"])

    cap.release()
    os.remove(video_path)

    avg_score = float(np.mean(scores)) if scores else 0.0

    verdict = (
        "Likely Fake" if avg_score > 0.7
        else "Suspicious" if avg_score > 0.4
        else "Real"
    )

    return {
        "status": "completed",
        "verdict": verdict,
        "confidence": round(avg_score, 4),
        "frames_analyzed": frame_count
    }


@celery_app.task(name="backend.tasks.analyze_video_task")
def analyze_video_task(video_bytes: bytes):
    return analyze_video_core(video_bytes)
