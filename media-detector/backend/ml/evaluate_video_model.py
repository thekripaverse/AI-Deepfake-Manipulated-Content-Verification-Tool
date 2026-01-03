import os
import sys
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ✅ Ensure project root is in PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.tasks.video_tasks import analyze_video_core


# ===============================
# CONFIGURATION
# ===============================

REAL_DIR = "backend/ml/eval_data/real"
FAKE_DIR = "backend/ml/eval_data/fake"

# Keep this low for faster evaluation
MAX_VIDEOS_PER_CLASS = 10   # change to 15/20 if you want
CONFIDENCE_THRESHOLD = 0.5


# ===============================
# DATA LOADING
# ===============================

def load_videos(folder: str, label: int, max_count: int):
    videos, labels = [], []

    files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
    files = files[:max_count]

    for file in files:
        path = os.path.join(folder, file)
        with open(path, "rb") as f:
            videos.append(f.read())
            labels.append(label)

    return videos, labels


# ===============================
# EVALUATION
# ===============================

def evaluate():
    print("\n🔹 Loading evaluation videos...")

    real_videos, real_labels = load_videos(
        REAL_DIR, label=0, max_count=MAX_VIDEOS_PER_CLASS
    )
    fake_videos, fake_labels = load_videos(
        FAKE_DIR, label=1, max_count=MAX_VIDEOS_PER_CLASS
    )

    videos = real_videos + fake_videos
    y_true = np.array(real_labels + fake_labels)
    y_scores = []

    print(f"✅ Loaded {len(real_videos)} real and {len(fake_videos)} fake videos\n")

    # ===============================
    # INFERENCE LOOP
    # ===============================

    for idx, video in enumerate(videos):
        print(f"▶ Processing video {idx + 1}/{len(videos)} ...")
        result = analyze_video_core(video)
        y_scores.append(result["confidence"])

    y_scores = np.array(y_scores)
    y_pred = (y_scores >= CONFIDENCE_THRESHOLD).astype(int)

    # ===============================
    # METRICS
    # ===============================

    print("\n===============================")
    print("📊 VIDEO MODEL PERFORMANCE")
    print("===============================")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC   : {roc_auc_score(y_true, y_scores):.4f}")
    print("===============================\n")


# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":
    evaluate()
