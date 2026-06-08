"""
Vrednotenje verifikacije obraza in optimizacija praga.

Postopek (posteno vrednotenje z loceno testno mnozico):
1. iz data/dataset/ osebe razdelimo na ucno/validacijsko/testno mnozico
   PO OSEBAH (app.dataset.train_val_test_split), da ista identiteta ni
   hkrati v validacijski in testni mnozici,
2. iz vsake mnozice zgradimo pozitivne in negativne pare,
3. vsak par predobdelamo in ocenimo z modelom (score_faces),
4. prag (hiperparameter match_threshold) OPTIMIZIRAMO na VALIDACIJSKI
   mnozici (preiscemo prage 0..1, izberemo najvisji F1),
5. izbrani prag UPORABIMO na TESTNI mnozici in porocamo metrike na njej.

Tako prag ni prilagojen istim podatkom, na katerih porocamo rezultate.

Uporaba:
    python -m scripts.evaluate --dataset data/dataset
"""

import argparse
from pathlib import Path

import cv2

from app.config import DEFAULT_CONFIG
from app.dataset import list_persons, make_pairs, train_val_test_split
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


def _score_pairs(pairs: list[tuple[Path, Path, int]]) -> list[tuple[float, int]]:
    scored: list[tuple[float, int]] = []
    for path_a, path_b, label in pairs:
        score = _score_pair(path_a, path_b)
        if score is not None:
            scored.append((score, label))
    return scored


def _best_threshold(scored: list[tuple[float, int]], steps: int) -> dict:
    """Poisci prag z najvisjim F1 na danih (ocenah, oznakah)."""
    y_true = [label for _, label in scored]
    best = {"threshold": 0.0, "f1": -1.0}
    for i in range(steps):
        threshold = i / (steps - 1)
        y_pred = [1 if score >= threshold else 0 for score, _ in scored]
        metrics = compute_metrics(y_true, y_pred)
        if metrics.f1 > best["f1"]:
            best = {"threshold": round(threshold, 3), "f1": metrics.f1}
    return best


def _metrics_at(scored: list[tuple[float, int]], threshold: float):
    y_true = [label for _, label in scored]
    y_pred = [1 if score >= threshold else 0 for score, _ in scored]
    return compute_metrics(y_true, y_pred)


def evaluate(dataset_dir: str, steps: int = 101, seed: int = 42) -> dict:
    persons = list_persons(dataset_dir)
    splits = train_val_test_split(persons, seed=seed)

    val_scored = _score_pairs(make_pairs(splits["val"], seed=seed))
    test_scored = _score_pairs(make_pairs(splits["test"], seed=seed))

    if not val_scored:
        raise RuntimeError(
            "Ni veljavnih validacijskih parov. Je data/dataset napolnjen "
            "z dovolj osebami? (potrebnih je vec oseb za smiselno delitev)"
        )

    best = _best_threshold(val_scored, steps)

    note = None
    if test_scored:
        test_metrics = _metrics_at(test_scored, best["threshold"])
    else:
        # premalo oseb za locen test -> porocamo na validaciji (z opozorilom)
        test_metrics = _metrics_at(val_scored, best["threshold"])
        note = ("Premalo oseb za loceno testno mnozico; metrike porocane na "
                "validacijski mnozici.")

    return {
        "threshold": best["threshold"],
        "val_pairs": len(val_scored),
        "test_pairs": len(test_scored),
        "metrics": test_metrics,
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate face verification")
    parser.add_argument("--dataset", default="data/dataset")
    parser.add_argument("--steps", type=int, default=101)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = evaluate(args.dataset, args.steps, args.seed)
    print(f"Val pairs (tuning)  : {result['val_pairs']}")
    print(f"Test pairs (report) : {result['test_pairs']}")
    print(f"Best threshold (val): {result['threshold']}")
    m = result["metrics"]
    print(f"Accuracy  : {m.accuracy}")
    print(f"Precision : {m.precision}")
    print(f"Recall    : {m.recall}")
    print(f"F1        : {m.f1}")
    print(f"Confusion (TP/FP/TN/FN): {m.tp}/{m.fp}/{m.tn}/{m.fn}")
    if result["note"]:
        print(f"Opomba: {result['note']}")


if __name__ == "__main__":
    main()
