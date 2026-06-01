"""
Verifikacija obraza s kombinacijo LBP histograma in ORB ujemanja.

Model ne uci utezi iz podatkov v klasicnem smislu, ampak zdruzi dva
komplementarna postopka racunalniskega vida:
- LBP histogram (globalna tekstura obraza) -> chi-square podobnost,
- ORB kljucne tocke (lokalna geometrija) -> delez dobrih ujemanj.

Koncni rezultat je utezena vsota obeh, odlocitev pa primerjava s pragom.
"""

from pathlib import Path

import cv2
import numpy as np

from app.config import ModelConfig, DEFAULT_CONFIG
from app.lbp import lbp_histogram
from app.orb_matching import orb_similarity
from app.similarity import chi_square_similarity
from app.image_preprocessing import bytes_to_cv_image, preprocess_for_model

DEBUG_DIR = Path("debug_images")
DEBUG_DIR.mkdir(exist_ok=True)


def score_faces(gray_a: np.ndarray, gray_b: np.ndarray,
                config: ModelConfig = DEFAULT_CONFIG) -> dict:
    """Izracunaj signale podobnosti dveh predobdelanih sivinskih obrazov.

    Vrne slovar z 'lbp_similarity', 'orb_similarity' in zdruzenim 'score'
    (utezena vsota v obmocju [0, 1]).
    """
    hist_a = lbp_histogram(gray_a, config.lbp_grid)
    hist_b = lbp_histogram(gray_b, config.lbp_grid)
    lbp_sim = chi_square_similarity(hist_a, hist_b)

    orb_sim = orb_similarity(
        gray_a, gray_b,
        n_features=config.orb_features,
        ratio=config.orb_ratio,
    )

    score = config.lbp_weight * lbp_sim + config.orb_weight * orb_sim

    return {
        "lbp_similarity": round(float(lbp_sim), 4),
        "orb_similarity": round(float(orb_sim), 4),
        "score": round(float(score), 4),
    }


def face_comparison(reference_image: bytes, current_image: bytes) -> dict:
    # TODO: povezano v naslednjem koraku (odlocitev + obravnava napak)
    reference_cv_image = bytes_to_cv_image(reference_image)
    current_cv_image = bytes_to_cv_image(current_image)
    return {
        "match": True,
        "confidence": 0.87,
        "message": "Fixed response. Face comparison is not implemented yet.",
    }
