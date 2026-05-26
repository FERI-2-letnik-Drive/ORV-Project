from pathlib import Path
import cv2
import numpy as np

DEBUG_DIR = Path("debug_images")
DEBUG_DIR.mkdir(exist_ok=True)

def bytes_to_cv_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Image bytes can't be empty")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR) # load as color image
    if image is None:
        raise ValueError("Image bytes can't be decoded")

    #print("Decoded image shape:", image.shape)

    #cv2.imwrite(str(DEBUG_DIR / f"{name}.jpg"), image)
    return image


def preprocess_image_for_comparison(image: np.ndarray) -> np.ndarray:
    image = crop_largest_face(image)
    image = cv2.resize(image, (256, 256))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if model requires uncomment
    image = image.astype(np.float32) / 255.0

    return image

def crop_largest_face(image: np.ndarray) -> np.ndarray:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.10, # increase search window size by 10%
        minNeighbors=5, # 5 is ok, change this to a higher value if it detects other faces
        minSize=(60, 60),
    )

    if len(faces) == 0:
        raise ValueError("No face detected")

    # choose the biggest detected face. #face[2] = width, face[3] height. Choose the largest area (face[2] * face[3])
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    # crop
    return image[y:y + h, x:x + w]

