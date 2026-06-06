"""
Konfiguracija modela za verifikacijo obraza.

Vse nastavljive vrednosti zdruzene na enem mestu, da jih je mogoce
spreminjati in optimizirati (npr. prag dolocimo s skripto za vrednotenje).

Koncni rezultat je utezena vsota dveh signalov:
    score = LBP_WEIGHT * lbp_similarity + ORB_WEIGHT * orb_similarity
Odlocitev o ujemanju: score >= MATCH_THRESHOLD.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # velikost obraza po predobdelavi (size x size)
    image_size: int = 256
    # mreza celic za LBP histogram
    lbp_grid: tuple[int, int] = (8, 8)
    # utezi obeh signalov (skupaj 1.0)
    lbp_weight: float = 0.6
    orb_weight: float = 0.4
    # eksponentna lestvica za LBP podobnost (manjsi = strozji)
    lbp_scale: float = 0.2
    # prag za odlocitev o ujemanju (kalibriran; optimiziran v evalvaciji)
    match_threshold: float = 0.45
    # ORB parametri
    orb_features: int = 500
    orb_ratio: float = 0.75


# privzeta konfiguracija, ki jo uporablja aplikacija
DEFAULT_CONFIG = ModelConfig()
