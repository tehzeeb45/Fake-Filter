# DeepFake Detection System - Core Code Snippet for SRS

---

## 1. File Validation & Decision Logic

```python
# Module: core/validator.py
def validate_uploaded_file(file_path: Path, file_size_mb: float, is_video: bool):
    """
    Validation Steps:
    1. Check file extension (.jpg, .jpeg, .png, .mp4, .avi).
    2. Check file size (Image <= 10 MB, Video <= 100 MB).
    3. Run face detection (MTCNN).
    """
    allowed_extensions = {".jpg", ".jpeg", ".png", ".mp4", ".avi"}
    extension = file_path.suffix.lower()

    # Step 1: File Extension Check
    if extension not in allowed_extensions:
        return {"valid": False, "error": "Unsupported file format"}

    # Step 2: File Size Check
    max_allowed_mb = 100.0 if is_video else 10.0
    if file_size_mb > max_allowed_mb:
        return {"valid": False, "error": "File exceeds size limit"}

    # Step 3: MTCNN Face Detection Check
    face_count = detect_faces_mtcnn(file_path)
    if face_count == 0:
        return {"valid": False, "error": "No face detected in media"}

    # All validation checks passed successfully
    return {"valid": True, "error": None}
```
