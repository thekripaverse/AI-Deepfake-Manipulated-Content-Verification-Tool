import cv2
import os
import tempfile
import numpy as np

from backend.ml.image_model import analyze_image
from PIL import Image


def extract_frames(video_path, fps=1):
    """
    Extract frames at given FPS
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps == 0:
        raise ValueError("Invalid video FPS")

    frame_interval = int(video_fps / fps)
    frames = []

    idx = 0
    success, frame = cap.read()

    while success:
        if idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

        success, frame = cap.read()
        idx += 1

    cap.release()
    return frames


def analyze_video(video_bytes: bytes):
    """
    Main video analysis pipeline
    """

    # ----------------------------
    # Save video temporarily
    # ----------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name

    try:
        frames = extract_frames(video_path, fps=1)

        if len(frames) == 0:
            raise ValueError("No frames extracted")

        scores = []

        for frame in frames:
            result = analyze_image(frame)
            scores.append(result["confidence"])

        avg_score = float(np.mean(scores))
        max_score = float(np.max(scores))

        # Conservative aggregation
        final_score = 0.6 * max_score + 0.4 * avg_score

        if final_score > 0.75:
            verdict = "Likely Fake"
        elif final_score > 0.45:
            verdict = "Suspicious"
        else:
            verdict = "Real"

        return {
            "verdict": verdict,
            "confidence": round(final_score, 4),
            "frames_analyzed": len(frames)
        }

    finally:
        # ----------------------------
        # Privacy: delete video
        # ----------------------------
        os.remove(video_path)
