"""
Verifikacija obraza s kombinacijo LBP histograma in ORB ujemanja.

Model ne uci utezi iz podatkov v klasicnem smislu, ampak zdruzi dva
komplementarna postopka racunalniskega vida:
- LBP histogram (globalna tekstura obraza) -> chi-square podobnost,
- ORB kljucne tocke (lokalna geometrija) -> delez dobrih ujemanj.

Koncni rezultat je utezena vsota obeh, odlocitev pa primerjava s pragom.
"""

import numpy as np

from app.config import ModelConfig, DEFAULT_CONFIG
from app.lbp import lbp_histogram
from app.orb_matching import orb_similarity
from app.similarity import exp_chi_square_similarity
from app.image_preprocessing import bytes_to_cv_image, preprocess_for_model


def score_faces(gray_a: np.ndarray, gray_b: np.ndarray,
                config: ModelConfig = DEFAULT_CONFIG) -> dict:
    """Izracunaj signale podobnosti dveh predobdelanih sivinskih obrazov.

    Vrne slovar z 'lbp_similarity', 'orb_similarity' in zdruzenim 'score'
    (utezena vsota v obmocju [0, 1]).
    """
    hist_a = lbp_histogram(gray_a, config.lbp_grid)
    hist_b = lbp_histogram(gray_b, config.lbp_grid)
    n_cells = config.lbp_grid[0] * config.lbp_grid[1]
    lbp_sim = exp_chi_square_similarity(hist_a, hist_b, n_cells, config.lbp_scale)

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


def face_comparison(reference_image: bytes, current_image: bytes,
                    config: ModelConfig = DEFAULT_CONFIG) -> dict:
    """Primerjaj referencni in trenutni obraz ter vrni odlocitev.

    Vrne slovar z 'match' (bool), 'confidence' (zdruzeni rezultat) in
    'message'. Ce na kateri sliki ni zaznan obraz, vrne match=False z
    razlago namesto napake.
    """
    reference_cv_image = bytes_to_cv_image(reference_image)
    current_cv_image = bytes_to_cv_image(current_image)

    try:
        gray_reference = preprocess_for_model(reference_cv_image, config.image_size)
        gray_current = preprocess_for_model(current_cv_image, config.image_size)
    except ValueError as error:
        # npr. "No face detected" iz izreza obraza
        return {
            "match": False,
            "confidence": 0.0,
            "message": f"Could not process face: {error}",
        }

    result = score_faces(gray_reference, gray_current, config)
    is_match = result["score"] >= config.match_threshold

    return {
        "match": bool(is_match),
        "confidence": result["score"],
        "message": (
            f"LBP={result['lbp_similarity']}, ORB={result['orb_similarity']}, "
            f"threshold={config.match_threshold}"
        ),
    }
