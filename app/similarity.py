"""
Mere podobnosti / razdalje med LBP histogrami.

Histograma sta L1-normalizirana znacilna vektorja. Za primerjavo dveh
obrazov potrebujemo mero, ki pove, kako podobna sta si histograma.
Uporabljamo dve klasicni meri:

- chi-square razdalja: obcutljiva na razlike v posameznih koshih, standard
  za primerjavo histogramov (manjsa = bolj podobno),
- kosinusna podobnost: kot med vektorjema (1 = enako, 0 = pravokotno).

Obe pretvorimo v podobnost v obmocju [0, 1], da ju je mogoce zdruzevati.
"""

import numpy as np

_EPS = 1e-10


def chi_square_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Chi-square razdalja med histogramoma (>= 0, manjsa = bolj podobno)."""
    a = hist_a.astype(np.float64)
    b = hist_b.astype(np.float64)
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + _EPS)))


def cosine_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Kosinusna podobnost med vektorjema (-1..1, za histograme 0..1)."""
    a = hist_a.astype(np.float64)
    b = hist_b.astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + _EPS
    return float(np.dot(a, b) / denom)


def chi_square_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Chi-square pretvorjena v podobnost [0, 1] (1 = identicna histograma)."""
    distance = chi_square_distance(hist_a, hist_b)
    return float(1.0 / (1.0 + distance))


def exp_chi_square_similarity(hist_a: np.ndarray, hist_b: np.ndarray,
                             n_cells: int, scale: float = 0.2) -> float:
    """Eksponentno skalirana chi-square podobnost [0, 1].

    Pri visokodimenzionalnih (mreznih) histogramih je surova chi-square
    razdalja velika, zato 1/(1+d) stisne vse rezultate proti 0. Namesto
    tega vzamemo POVPRECNO razdaljo na celico (d / n_cells) in jo skozi
    exp(-d/scale) preslikamo v [0, 1]. To da bolje locljive rezultate:
    identicna histograma -> 1.0, razlicna -> proti 0.

    'scale' uravnava obcutljivost (manjsi = strozji).
    """
    mean_distance = chi_square_distance(hist_a, hist_b) / max(n_cells, 1)
    return float(np.exp(-mean_distance / scale))
