from pathlib import Path
import cv2
import numpy as np

from app.image_preprocessing import (
    bytes_to_cv_image,
    preprocess_image_for_comparison,
)

DEBUG_DIR = Path("debug_images")
DEBUG_DIR.mkdir(exist_ok=True)

def face_comparison(reference_image: bytes, current_image: bytes) -> dict:
    reference_cv_image = bytes_to_cv_image(reference_image) #"reference_image"
    current_cv_image = bytes_to_cv_image(current_image) # "current_image"

    processed_reference_image = preprocess_image_for_comparison(reference_cv_image)
    processed_current_image = preprocess_image_for_comparison(current_cv_image)

    cv2.imwrite(str(DEBUG_DIR / "reference.jpg"), (processed_reference_image * 255).astype(np.uint8))
    cv2.imwrite(str(DEBUG_DIR / "current.jpg"), (processed_current_image * 255).astype(np.uint8))

    return {
        "match": True,
        "confidence": 0.87,
        "message": "Fixed response. Face comparison is not implemented yet."
    }