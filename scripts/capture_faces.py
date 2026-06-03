"""
Zajem ucnih slik obraza z webkamere.

Uporaba:
    python -m scripts.capture_faces --person alice --count 20

Skripta odpre kamero, sproti zaznava obraz (Haar kaskada) in ob pritisku
na SPACE shrani trenutni okvir v data/dataset/<person>/. Tipka Q konca.
Slike se shranjujejo oznacene zaporedno (alice_000.jpg, alice_001.jpg ...).

To je orodje clana, ki skrbi za zajem in pripravo podatkov. Zajem zahteva
kamero, zato se ne izvaja v testih.
"""

import argparse
from pathlib import Path

import cv2

DATASET_DIR = Path("data/dataset")


def capture(person: str, count: int = 20, camera: int = 0) -> Path:
    out_dir = DATASET_DIR / person
    out_dir.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    saved = 0
    print("SPACE = shrani sliko, Q = koncaj")
    while saved < count:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"{person}: {saved}/{count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and len(faces) > 0:
            path = out_dir / f"{person}_{saved:03d}.jpg"
            cv2.imwrite(str(path), frame)
            print("saved", path)
            saved += 1

    cap.release()
    cv2.destroyAllWindows()
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture face images from webcam")
    parser.add_argument("--person", required=True, help="ime osebe / oznaka mape")
    parser.add_argument("--count", type=int, default=20, help="stevilo slik")
    parser.add_argument("--camera", type=int, default=0, help="indeks kamere")
    args = parser.parse_args()
    capture(args.person, args.count, args.camera)


if __name__ == "__main__":
    main()
