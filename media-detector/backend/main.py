from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from backend.api.media import router as media_router
from backend.utils.rate_limit import limiter
app = FastAPI(
    title="AI Deepfake & Manipulated Content Verification Tool",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(media_router, prefix="/media")
@app.get("/")
def root():
    return {
        "status": "AI Deepfake Detection API running",
        "privacy": "No media is stored",
        "features": [
            "Deepfake image detection",
            "Explainable AI (Grad-CAM)",
            "Rate limiting",
            "In-memory processing"
        ]
    }
