import pytest

from app.metrics import confusion_counts, compute_metrics


def test_confusion_counts_basic():
    tp, fp, tn, fn = confusion_counts([1, 1, 0, 0], [1, 0, 0, 1])
    assert (tp, fp, tn, fn) == (1, 1, 1, 1)


def test_perfect_prediction():
    m = compute_metrics([1, 0, 1, 0], [1, 0, 1, 0])
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_all_wrong_prediction():
    m = compute_metrics([1, 1, 0, 0], [0, 0, 1, 1])
    assert m.accuracy == 0.0
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_metrics_values():
    m = compute_metrics([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert m.accuracy == 0.6
    assert m.precision == round(2 / 3, 4)
    assert m.recall == round(2 / 3, 4)


def test_no_positive_predictions_precision_zero():
    m = compute_metrics([1, 0], [0, 0])
    assert m.precision == 0.0
    assert m.recall == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_counts([1, 0], [1])
