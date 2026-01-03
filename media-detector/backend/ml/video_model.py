import cv2
import os
import tempfile
import numpy as np
from backend.ml.image_model import analyze_image
from PIL import Image


def extract_frames(video_path, max_frames=12):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    idxs = np.linspace(0, total_frames - 1, max_frames).astype(int)
    frames = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def analyze_video(video_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name

    try:
        frames = extract_frames(video_path)

        if len(frames) == 0:
            raise ValueError("No frames extracted")

        scores = []
        for frame in frames:
            result = analyze_image(frame)
            scores.append(result["confidence"])

        scores = np.array(scores)

        top_k = max(2, int(0.3 * len(scores)))
        suspicious_score = float(np.mean(np.sort(scores)[-top_k:]))

        if suspicious_score >= 0.65:
            verdict = "Likely Fake"
        elif suspicious_score >= 0.45 and suspicious_score < 0.65:
            verdict = "Suspicious"
        else:
            verdict = "Real"

        return {
            "verdict": verdict,
            "confidence": round(float(suspicious_score), 4),
            "frames_analyzed": len(frames)
        }

    finally:
        os.remove(video_path)
