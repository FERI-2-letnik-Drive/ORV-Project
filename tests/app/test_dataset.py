from pathlib import Path

import pytest

from app.dataset import list_persons, make_pairs, train_val_test_split


def _make_fake_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for person, n in {"alice": 3, "bob": 2, "carol": 2}.items():
        d = root / person
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{person}_{i}.jpg").write_bytes(b"fake")
    return root


def test_list_persons_finds_images(tmp_path):
    root = _make_fake_dataset(tmp_path)
    persons = list_persons(root)
    assert set(persons.keys()) == {"alice", "bob", "carol"}
    assert len(persons["alice"]) == 3


def test_list_persons_missing_dir_returns_empty():
    assert list_persons("does/not/exist") == {}


def test_make_pairs_has_both_labels_and_valid_negatives(tmp_path):
    persons = list_persons(_make_fake_dataset(tmp_path))
    pairs = make_pairs(persons, seed=1)
    labels = [label for _, _, label in pairs]
    assert 1 in labels and 0 in labels
    # vsak negativen par mora biti iz razlicnih map
    for a, b, label in pairs:
        if label == 0:
            assert a.parent.name != b.parent.name


def test_make_pairs_positive_count_is_deterministic(tmp_path):
    persons = list_persons(_make_fake_dataset(tmp_path))
    # alice: C(3,2)=3, bob: 1, carol: 1 -> 5 pozitivnih
    positives = [p for p in make_pairs(persons) if p[2] == 1]
    assert len(positives) == 5


def test_split_is_identity_disjoint_and_full():
    persons = {f"p{i}": [Path("x.jpg")] for i in range(10)}
    splits = train_val_test_split(persons, (0.6, 0.2, 0.2))
    names = [set(splits[s].keys()) for s in ("train", "val", "test")]
    assert names[0].isdisjoint(names[1])
    assert names[0].isdisjoint(names[2])
    assert names[1].isdisjoint(names[2])
    assert set().union(*names) == set(persons.keys())


def test_split_invalid_ratios_raises():
    with pytest.raises(ValueError):
        train_val_test_split({"a": []}, (0.5, 0.2, 0.2))
