"""
ORB ujemanje znacilk kot sekundarni signal podobnosti obrazov.

LBP histogram opise globalno teksturo obraza, ORB (Oriented FAST and
Rotated BRIEF) pa poisce lokalne kljucne tocke (koti, robovi) in jih
opise z binarnimi deskriptorji. Ce dva obraza pripadata isti osebi, se
veliko kljucnih tock dobro ujema.

Postopek:
1. ORB poisce kljucne tocke in deskriptorje na obeh sivinskih slikah,
2. BFMatcher (Hamming razdalja) za vsako tocko poisce 2 najblizja para,
3. Lowejev ratio test obdrzi le zanesljive pare (najblizji je obcutno
   blizje od drugega najblizjega),
4. podobnost = delez dobrih ujemanj glede na stevilo kljucnih tock.
"""

import cv2
import numpy as np


def orb_similarity(gray_a: np.ndarray, gray_b: np.ndarray,
                   n_features: int = 500, ratio: float = 0.75) -> float:
    """Vrne ORB podobnost dveh sivinskih slik v obmocju [0, 1].

    0 pomeni brez ujemanj (ali premalo kljucnih tock), visje = bolj podobno.
    """
    orb = cv2.ORB_create(nfeatures=n_features)

    keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(gray_b, None)

    # ce na kateri od slik ni dovolj tock, ne moremo primerjati
    if descriptors_a is None or descriptors_b is None:
        return 0.0
    if len(keypoints_a) < 2 or len(keypoints_b) < 2:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)

    good = 0
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        nearest, second = pair
        # Lowejev ratio test: obdrzimo le izrazito boljsa ujemanja
        if nearest.distance < ratio * second.distance:
            good += 1

    # normaliziramo z manjsim stevilom tock (najvec mozno ujemanj)
    denom = min(len(keypoints_a), len(keypoints_b))
    if denom == 0:
        return 0.0
    return float(good / denom)
