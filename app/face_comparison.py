import cv2
import numpy as np

def bytes_to_cv_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Image bytes can't be empty")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR) # load as color image
    if image is None:
        raise ValueError("Image bytes can't be decoded")

    return image


def face_comparison(reference_image: bytes, current_image: bytes) -> dict:
    reference_cv_image = bytes_to_cv_image(reference_image)
    current_cv_image = bytes_to_cv_image(current_image)

    return {
        "match": True,
        "confidence": 0.87,
        "message": "Fixed response. Face comparison is not implemented yet."
    }