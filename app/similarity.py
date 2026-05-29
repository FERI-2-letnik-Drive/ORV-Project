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
