"""
Organizacija dataseta in priprava parov za vrednotenje verifikacije.

Pricakovana struktura map:
    data/dataset/
        alice/  alice_000.jpg, alice_001.jpg, ...
        bob/    bob_000.jpg, ...

Za verifikacijo obraza potrebujemo PARE slik z oznako:
- pozitiven par (label 1): dve sliki iste osebe,
- negativen par (label 0): sliki dveh razlicnih oseb.
"""

import random
from itertools import combinations
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_persons(dataset_dir: str | Path) -> dict[str, list[Path]]:
    """Vrne slovar {oseba: [poti do slik]} iz strukture map."""
    dataset_dir = Path(dataset_dir)
    persons: dict[str, list[Path]] = {}
    if not dataset_dir.exists():
        return persons
    for person_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        images = sorted(
            img for img in person_dir.iterdir()
            if img.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            persons[person_dir.name] = images
    return persons


def make_pairs(persons: dict[str, list[Path]],
               max_negatives: int | None = None,
               seed: int = 42) -> list[tuple[Path, Path, int]]:
    """Zgradi pozitivne in negativne pare.

    Pozitivni pari: vse kombinacije slik znotraj iste osebe (label 1).
    Negativni pari: nakljucni pari slik razlicnih oseb (label 0).
    Stevilo negativov uravnotezimo s pozitivi (oz. omejimo z max_negatives).
    """
    rng = random.Random(seed)

    positives: list[tuple[Path, Path, int]] = []
    for images in persons.values():
        for a, b in combinations(images, 2):
            positives.append((a, b, 1))

    # vse slike z oznako osebe za generiranje negativov
    flat = [(name, img) for name, images in persons.items() for img in images]
    target_negatives = max_negatives if max_negatives is not None else len(positives)

    negatives: list[tuple[Path, Path, int]] = []
    attempts = 0
    while len(negatives) < target_negatives and attempts < target_negatives * 20 + 100:
        attempts += 1
        (name_a, img_a), (name_b, img_b) = rng.sample(flat, 2) if len(flat) >= 2 else (None, None)
        if name_a != name_b:
            negatives.append((img_a, img_b, 0))

    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


def train_val_test_split(persons: dict[str, list[Path]],
                         ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                         seed: int = 42) -> dict[str, dict[str, list[Path]]]:
    """Razdeli osebe v ucno, validacijsko in testno mnozico.

    Delitev je po OSEBAH (ne po slikah), da ista identiteta ne pride v vec
    mnozic hkrati - to je nujno za posteno vrednotenje verifikacije.

    Vrne {'train': {...}, 'val': {...}, 'test': {...}}.
    """
    if not abs(sum(ratios) - 1.0) < 1e-6:
        raise ValueError("ratios must sum to 1.0")

    names = list(persons.keys())
    rng = random.Random(seed)
    rng.shuffle(names)

    n = len(names)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        "train": names[:n_train],
        "val": names[n_train:n_train + n_val],
        "test": names[n_train + n_val:],
    }
    return {
        split: {name: persons[name] for name in split_names}
        for split, split_names in splits.items()
    }
