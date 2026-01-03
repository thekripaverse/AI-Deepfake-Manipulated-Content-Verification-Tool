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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        os.remove(video_path)
        return {"status": "error", "confidence": 0.0}

    idxs = np.linspace(0, total_frames - 1, 12).astype(int)
    scores = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        result = analyze_image(frame_pil)
        scores.append(result["confidence"])

    cap.release()
    os.remove(video_path)

    scores = np.array(scores)

    if len(scores) == 0:
        return {"status": "error", "confidence": 0.0}

    suspicious_score = np.percentile(scores, 75)

    verdict = (
        "Likely Fake" if suspicious_score > 0.60
        else "Suspicious" if suspicious_score > 0.40
        else "Real"
    )

    return {
        "status": "completed",
        "verdict": verdict,
        "confidence": round(float(suspicious_score), 4),
        "frames_analyzed": len(scores)
    }


@celery_app.task(name="backend.tasks.analyze_video_task")
def analyze_video_task(video_bytes: bytes):
    return analyze_video_core(video_bytes)
