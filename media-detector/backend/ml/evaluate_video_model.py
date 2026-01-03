import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.tasks.video_tasks import analyze_video_core

REAL_DIR = "backend/ml/eval_data/real"
FAKE_DIR = "backend/ml/eval_data/fake"

MAX_VIDEOS_PER_CLASS = 10
CONFIDENCE_THRESHOLD = 0.40   # 🔑 recall-friendly


def load_videos(folder, label, max_count):
    videos, labels = [], []
    files = [f for f in os.listdir(folder) if f.endswith(".mp4")][:max_count]

    for f in files:
        with open(os.path.join(folder, f), "rb") as vid:
            videos.append(vid.read())
            labels.append(label)

    return videos, labels


def evaluate():
    print("\n🔹 Loading evaluation videos...")

    real_videos, real_labels = load_videos(REAL_DIR, 0, MAX_VIDEOS_PER_CLASS)
    fake_videos, fake_labels = load_videos(FAKE_DIR, 1, MAX_VIDEOS_PER_CLASS)

    videos = real_videos + fake_videos
    y_true = np.array(real_labels + fake_labels)
    y_scores = []

    for i, video in enumerate(videos):
        print(f"▶ Processing video {i + 1}/{len(videos)}")
        result = analyze_video_core(video)
        y_scores.append(result["confidence"])

    y_scores = np.array(y_scores)
    y_pred = (y_scores >= CONFIDENCE_THRESHOLD).astype(int)
    print("-- VIDEO MODEL PERFORMANCE")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred):.4f}")

    if len(np.unique(y_true)) > 1:
        print(f"ROC AUC   : {roc_auc_score(y_true, y_scores):.4f}")
if __name__ == "__main__":
    evaluate()
