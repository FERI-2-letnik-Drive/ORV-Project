"""
Local Binary Pattern (LBP) operator.

LBP je klasicni postopek racunalniskega vida za opis teksture. Za vsak
slikovni piksel pogledamo 8 sosedov v krogu polmera 1. Ce je sosed
svetlejsi ali enak srediscu, zapisemo 1, sicer 0. Iz 8 bitov sestavimo
8-bitno stevilo (0-255), ki opisuje lokalni vzorec teksture okoli piksla.

Implementirano lastnorocno (numpy), brez zunanjih LBP knjiznic, da je
postopek razumljiv in prilagodljiv.
"""

import numpy as np


def compute_lbp(gray: np.ndarray) -> np.ndarray:
    """Izracunaj LBP sliko iz sivinske slike (uint8, 2D).

    Vrne uint8 sliko enakih dimenzij, kjer vsak piksel hrani LBP kodo.
    Robni pikseli (1px okvir) se nastavijo na 0, ker nimajo vseh sosedov.
    """
    if gray.ndim != 2:
        raise ValueError("compute_lbp expects a 2D grayscale image")

    gray = gray.astype(np.int16)
    center = gray[1:-1, 1:-1]

    # 8 sosedov sredisca, vsak prispeva svojo utez (1, 2, 4, ... 128)
    neighbours = [
        (gray[0:-2, 0:-2], 1),    # zgoraj-levo
        (gray[0:-2, 1:-1], 2),    # zgoraj
        (gray[0:-2, 2:], 4),      # zgoraj-desno
        (gray[1:-1, 2:], 8),      # desno
        (gray[2:, 2:], 16),       # spodaj-desno
        (gray[2:, 1:-1], 32),     # spodaj
        (gray[2:, 0:-2], 64),     # spodaj-levo
        (gray[1:-1, 0:-2], 128),  # levo
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)
    for neighbour, weight in neighbours:
        lbp += ((neighbour >= center) * weight).astype(np.uint8)

    # vstavi nazaj v okvir velikosti originala
    out = np.zeros(gray.shape, dtype=np.uint8)
    out[1:-1, 1:-1] = lbp
    return out
