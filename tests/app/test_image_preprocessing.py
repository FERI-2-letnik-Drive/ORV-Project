import cv2
import numpy as np
import pytest

from app.image_preprocessing import (
    bytes_to_cv_image,
    preprocess_image_for_comparison,
    crop_largest_face,
)


def test_bytes_to_cv_image_empty_bytes():
    with pytest.raises(ValueError, match="Image bytes can't be empty"):
        bytes_to_cv_image(b"")


def test_bytes_to_cv_image_invalid_bytes():
    with pytest.raises(ValueError, match="Image bytes can't be decoded"):
        bytes_to_cv_image(b"not-an-image")


def test_preprocess_image_shape():
    image = cv2.imread("utils/man-face-test.jpg")

    processed = preprocess_image_for_comparison(image)

    assert processed.shape == (256, 256, 3)

'''
def test_preprocess_image_normalized():
    image = cv2.imread("utils/man-face-test.jpg")

    processed = preprocess_image_for_comparison(image)

    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
'''

def test_crop_largest_face_no_face():
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="No face detected"):
        crop_largest_face(image)