import cv2
import numpy as np

from app.orb_matching import orb_similarity


def _load_face_gray():
    return cv2.imread("utils/man-face-test.jpg", cv2.IMREAD_GRAYSCALE)


def test_orb_identical_image_is_high():
    face = _load_face_gray()
    assert orb_similarity(face, face) > 0.9


def test_orb_face_vs_noise_is_low():
    face = _load_face_gray()
    noise = (np.random.rand(*face.shape) * 255).astype(np.uint8)
    assert orb_similarity(face, noise) < 0.2


def test_orb_blank_images_return_zero():
    blank = np.zeros((128, 128), dtype=np.uint8)
    # na enobarvni sliki ORB ne najde kljucnih tock -> 0.0
    assert orb_similarity(blank, blank) == 0.0


def test_orb_similarity_in_unit_range():
    face = _load_face_gray()
    rotated = cv2.rotate(face, cv2.ROTATE_90_CLOCKWISE)
    score = orb_similarity(face, rotated)
    assert 0.0 <= score <= 1.0
