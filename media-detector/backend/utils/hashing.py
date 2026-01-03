import hashlib

def generate_file_hash(file_bytes: bytes) -> str:
    sha = hashlib.sha256()
    sha.update(file_bytes)
    return sha.hexdigest()
