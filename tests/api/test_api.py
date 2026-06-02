from fastapi.testclient import TestClient

from api.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_face_verification_missing_files():
    response = client.post("/api/v1/face-verifications")

    assert response.status_code == 422 # 422 Unprocessable Entity, Required data is missing

def test_face_verification_missing_current_image():
    files = {
        "reference_image": ("reference.jpg", b"fake-image", "image/jpeg"), # no actual file is created, it's a fake
    }

    response = client.post(
        "/api/v1/face-verifications",
        files=files
    )

    assert response.status_code == 422

def test_face_verification_success(monkeypatch):
    def fake_compare_faces(reference_image, current_image):
        return {
            "match": True,
            "confidence": 0.99,
            "message": "ok"
        }

    # temporarily replacing that function with our test function
    monkeypatch.setattr(
        "api.api.compare_faces",
        fake_compare_faces
    )

    files = {
        "reference_image": ("reference.jpg", b"fake", "image/jpeg"),
        "current_image": ("current.jpg", b"fake", "image/jpeg"),
    }

    response = client.post(
        "/api/v1/face-verifications",
        files=files
    )

    assert response.status_code == 200

    data = response.json()

    assert data["match"] is True
    assert data["confidence"] == 0.99
    assert data["message"] == "ok"

def _blank_jpg_bytes():
    import cv2
    import numpy as np
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", blank)
    return buffer.tobytes()


def test_face_verification_invalid_image_returns_400():
    files = {
        "reference_image": ("reference.jpg", b"not-an-image", "image/jpeg"),
        "current_image": ("current.jpg", b"not-an-image", "image/jpeg"),
    }
    response = client.post("/api/v1/face-verifications", files=files)
    assert response.status_code == 400


def test_face_verification_no_face_returns_match_false():
    blank = _blank_jpg_bytes()
    real = open("utils/man-face-test.jpg", "rb").read()
    files = {
        "reference_image": ("reference.jpg", blank, "image/jpeg"),
        "current_image": ("current.jpg", real, "image/jpeg"),
    }
    response = client.post("/api/v1/face-verifications", files=files)
    assert response.status_code == 200
    assert response.json()["match"] is False


def test_face_verification_same_person_matches():
    real = open("utils/man-face-test.jpg", "rb").read()
    files = {
        "reference_image": ("reference.jpg", real, "image/jpeg"),
        "current_image": ("current.jpg", real, "image/jpeg"),
    }
    response = client.post("/api/v1/face-verifications", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["match"] is True
    assert data["confidence"] >= 0.5
