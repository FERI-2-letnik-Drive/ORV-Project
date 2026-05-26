import cv2
import numpy as np

# flip horizontaly *.. -> ..*. 0 = top-bottom, -1 = both directions
def augment_flip(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)

# rotate around center, default 10 degrees. Positive angle -> counter-clockwise else clockwise
def augment_rotate(image: np.ndarray, angle: float = 10.0) -> np.ndarray:
    height, width = image.shape[:2]

    # where each pixel should move
    rotation_matrix = cv2.getRotationMatrix2D(
        (width // 2, height // 2),
        angle,
        1.0 # keep the same scale
    )

    # performs the rotation
    return cv2.warpAffine(image, rotation_matrix, (width, height))

# negative -> darker, positive -> lighter
def augment_brightness(image: np.ndarray, beta: int = 30) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)

# for model
def create_augmented_images(image: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "flipped": augment_flip(image),
        "rotated_right": augment_rotate(image, -10),
        "rotated_left": augment_rotate(image, 10),
        "brighter": augment_brightness(image, 30),
        "darker": augment_brightness(image, -30),
    }