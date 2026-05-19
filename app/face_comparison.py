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

def preprocess_image_for_comparison(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (256, 256))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if model requires uncomment
    image = image.astype(np.float32) / 255.0

    return image


def face_comparison(reference_image: bytes, current_image: bytes) -> dict:
    reference_cv_image = bytes_to_cv_image(reference_image)
    current_cv_image = bytes_to_cv_image(current_image)

    reference_image = preprocess_image_for_comparison(reference_cv_image)
    current_image = preprocess_image_for_comparison(current_cv_image)

    return {
        "match": True,
        "confidence": 0.87,
        "message": "Fixed response. Face comparison is not implemented yet."
    }