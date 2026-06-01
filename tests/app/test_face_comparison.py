import cv2
import numpy as np
import pytest

from app.face_comparison import score_faces, face_comparison
from app.image_preprocessing import preprocess_for_model


def _face_bytes():
    return open("utils/man-face-test.jpg", "rb").read()


def _blank_jpg_bytes():
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", blank)
    return buffer.tobytes()


def test_score_faces_identical_is_one():
    gray = preprocess_for_model(cv2.imread("utils/man-face-test.jpg"))
    result = score_faces(gray, gray)
    assert result["score"] == 1.0
    assert result["lbp_similarity"] == 1.0
    assert result["orb_similarity"] == 1.0


def test_score_faces_components_in_unit_range():
    gray = preprocess_for_model(cv2.imread("utils/man-face-test.jpg"))
    flipped = cv2.flip(gray, 1)
    result = score_faces(gray, flipped)
    for key in ("lbp_similarity", "orb_similarity", "score"):
        assert 0.0 <= result[key] <= 1.0


def test_face_comparison_same_person_matches():
    reference = _face_bytes()
    result = face_comparison(reference, reference)
    assert result["match"] is True
    assert result["confidence"] >= 0.5


def test_face_comparison_no_face_returns_false():
    result = face_comparison(_blank_jpg_bytes(), _face_bytes())
    assert result["match"] is False
    assert "face" in result["message"].lower()


def test_face_comparison_invalid_bytes_raises():
    with pytest.raises(ValueError):
        face_comparison(b"not-an-image", _face_bytes())
