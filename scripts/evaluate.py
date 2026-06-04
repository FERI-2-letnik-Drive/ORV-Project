"""
Vrednotenje verifikacije obraza in optimizacija praga.

Postopek:
1. iz data/dataset/ zgradi pozitivne in negativne pare (app.dataset),
2. vsak par predobdela in oceni z modelom (app.face_comparison.score_faces),
3. preisce prage od 0 do 1 in poisce tistega z najvisjim F1 (optimizacija
   hiperparametra 'match_threshold'),
4. izpise metrike (tocnost, preciznost, priklic, F1) pri najboljsem pragu.

Uporaba:
    python -m scripts.evaluate --dataset data/dataset
"""

import argparse
from pathlib import Path

import cv2

from app.config import DEFAULT_CONFIG
from app.dataset import list_persons, make_pairs
from app.face_comparison import score_faces
from app.image_preprocessing import preprocess_for_model
from app.metrics import compute_metrics


def _score_pair(path_a: Path, path_b: Path) -> float | None:
    image_a = cv2.imread(str(path_a))
    image_b = cv2.imread(str(path_b))
    if image_a is None or image_b is None:
        return None
    try:
        gray_a = preprocess_for_model(image_a, DEFAULT_CONFIG.image_size)
        gray_b = preprocess_for_model(image_b, DEFAULT_CONFIG.image_size)
    except ValueError:
        return None  # obraz ni zaznan -> par izpustimo
    return score_faces(gray_a, gray_b, DEFAULT_CONFIG)["score"]


def evaluate(dataset_dir: str, steps: int = 101) -> dict:
    persons = list_persons(dataset_dir)
    pairs = make_pairs(persons)

    scored: list[tuple[float, int]] = []
    for path_a, path_b, label in pairs:
        score = _score_pair(path_a, path_b)
        if score is not None:
            scored.append((score, label))

    if not scored:
        raise RuntimeError("No valid pairs scored. Is data/dataset populated?")

    y_true = [label for _, label in scored]

    best = {"threshold": 0.0, "f1": -1.0, "metrics": None}
    for i in range(steps):
        threshold = i / (steps - 1)
        y_pred = [1 if score >= threshold else 0 for score, _ in scored]
        metrics = compute_metrics(y_true, y_pred)
        if metrics.f1 > best["f1"]:
            best = {"threshold": round(threshold, 3), "f1": metrics.f1,
                    "metrics": metrics}

    return {"num_pairs": len(scored), **best}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate face verification")
    parser.add_argument("--dataset", default="data/dataset")
    parser.add_argument("--steps", type=int, default=101)
    args = parser.parse_args()

    result = evaluate(args.dataset, args.steps)
    print(f"Pairs evaluated : {result['num_pairs']}")
    print(f"Best threshold  : {result['threshold']}")
    m = result["metrics"]
    print(f"Accuracy        : {m.accuracy}")
    print(f"Precision       : {m.precision}")
    print(f"Recall          : {m.recall}")
    print(f"F1              : {m.f1}")
    print(f"Confusion (TP/FP/TN/FN): {m.tp}/{m.fp}/{m.tn}/{m.fn}")


if __name__ == "__main__":
    main()
