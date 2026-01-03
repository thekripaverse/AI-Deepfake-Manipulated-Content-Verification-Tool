import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
import numpy as np
import cv2
import base64

from backend.ml.gradcam import GradCAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained EfficientNet
weights = EfficientNet_B4_Weights.DEFAULT
model = efficientnet_b4(weights=weights)
model.classifier[1] = torch.nn.Linear(1792, 1)
model.eval().to(DEVICE)

# NOTE:
# In real deployment, replace this with a fine-tuned deepfake model
# trained on FaceForensics++ / Celeb-DF
gradcam = GradCAM(model, model.features[-1])

transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def analyze_image(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits).item()

    if prob > 0.75:
        verdict = "Likely Fake"
    elif prob > 0.45:
        verdict = "Suspicious"
    else:
        verdict = "Real"
    image_tensor.requires_grad_(True)
    with torch.enable_grad():
        logits_gc = model(image_tensor)
        class_score = logits_gc[0]
        cam_map = gradcam.generate(class_score)
        
    # Overlay heatmap
    img_np = np.array(image.resize((380, 380)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Encode to base64
    _, buffer = cv2.imencode(".png", overlay)
    heatmap_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "verdict": verdict,
        "confidence": round(prob, 4),
        "heatmap": heatmap_b64
    }

