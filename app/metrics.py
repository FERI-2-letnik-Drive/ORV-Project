"""
Klasifikacijske metrike za vrednotenje verifikacije obraza.

Verifikacija je binarna klasifikacija: par je 'isti' (1) ali 'razlicen' (0).
Model vrne rezultat (score); s pragom ga pretvorimo v napoved. Metrike
primerjajo napovedi z resnicnimi oznakami.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


def confusion_counts(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int, int]:
    """Vrne (TP, FP, TN, FN)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    tp = fp = tn = fn = 0
    for true, pred in zip(y_true, y_pred):
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def compute_metrics(y_true: list[int], y_pred: list[int]) -> Metrics:
    """Izracunaj tocnost, preciznost, priklic in F1."""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return Metrics(
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        tp=tp, fp=fp, tn=tn, fn=fn,
    )
