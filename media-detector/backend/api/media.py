from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from PIL import Image
import io
import imghdr

# Image analysis
from backend.ml.image_model import analyze_image

# Video async analysis
from backend.celery_app import celery_app

# Security & utilities
from backend.utils.hashing import generate_file_hash
from backend.utils.audit import audit_log
from backend.utils.rate_limit import limiter

router = APIRouter()
MAX_IMAGE_SIZE_MB = 10
MAX_VIDEO_SIZE_MB = 50
ALLOWED_IMAGE_TYPES = {"jpeg", "png"}
@router.post("/analyze/image")
@limiter.limit("5/minute")
async def analyze_image_api(
    request: Request,
    file: UploadFile = File(...)
):
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail="Image too large. Maximum size is 10MB."
        )
    image_type = imghdr.what(None, h=file_bytes)
    if image_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image format"
        )
    image_hash = generate_file_hash(file_bytes)
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Corrupted or unreadable image"
        )
    result = analyze_image(image)
    log_entry = audit_log("image_analyzed", image_hash)
    print(log_entry)
    del file_bytes
    del image

    return {
        "media_type": "image",
        "hash": image_hash,
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "explanation": "Highlighted regions indicate forensic inconsistencies",
        "heatmap_base64": result["heatmap"],
        "privacy": "Image processed in-memory and immediately discarded."
    }

@router.post("/analyze/video")
@limiter.limit("2/minute")
async def analyze_video_api(
    request: Request,
    file: UploadFile = File(...)
):
    video_bytes = await file.read()

    size_mb = len(video_bytes) / (1024 * 1024)
    if size_mb > 50:
        raise HTTPException(
            status_code=413,
            detail="Video too large. Maximum size is 50MB."
        )

    task = celery_app.send_task(
        "backend.tasks.analyze_video_task",
        args=[video_bytes]
    )

    del video_bytes

    return {
        "media_type": "video",
        "job_id": task.id,
        "status": "processing",
        "message": "Video analysis started asynchronously"
    }

@router.get("/status/{job_id}")
async def check_video_status(job_id: str):
    task = celery_app.AsyncResult(job_id)

    if task.state == "PENDING":
        return {
            "status": "pending",
            "message": "Video is still being processed"
        }

    if task.state == "FAILURE":
        return {
            "status": "failed",
            "error": str(task.result)
        }

    if task.state == "SUCCESS":
        return task.result

    return {
        "status": task.state
    }
