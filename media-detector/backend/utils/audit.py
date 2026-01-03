import time

def audit_log(event: str, media_hash: str):
    return {
        "event": event,
        "media_hash": media_hash,
        "timestamp": int(time.time())
    }
