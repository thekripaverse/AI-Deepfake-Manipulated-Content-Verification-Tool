from celery import Celery

celery_app = Celery(
    "deepfake_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# 🔑 Tell Celery EXACTLY where to look for tasks
celery_app.autodiscover_tasks(
    packages=["backend.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
