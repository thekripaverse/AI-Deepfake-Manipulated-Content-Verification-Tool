import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from image_model import analyze_image

TEST_DIR = "data/test"

y_true = []
y_pred = []
y_scores = []

def load_images(label, folder):
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png")):
            path = os.path.join(folder, file)
            img = Image.open(path).convert("RGB")
            result = analyze_image(img)

            score = result["confidence"]
            prediction = 1 if score > 0.5 else 0

            y_true.append(label)
            y_pred.append(prediction)
            y_scores.append(score)

# 0 = real, 1 = fake
load_images(0, os.path.join(TEST_DIR, "real"))
load_images(1, os.path.join(TEST_DIR, "fake"))

print("Evaluation Results")
print("------------------")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1-score :", f1_score(y_true, y_pred))
print("ROC-AUC  :", roc_auc_score(y_true, y_scores))
