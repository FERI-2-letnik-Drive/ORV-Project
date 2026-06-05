# Model: verifikacija obraza (LBP + ORB)

## Cilj

Ugotoviti, ali dve sliki prikazujeta isto osebo (binarna verifikacija).
Rešitev je namenoma klasična (brez globokih mrež), da je vsak korak
razumljiv in lasten.

## Cevovod

1. **Predobdelava** (`app/image_preprocessing.py`)
   - izrez največjega obraza (Haar kaskada),
   - sprememba velikosti na 256×256,
   - pretvorba v sivinski prostor,
   - odstranjevanje šuma (medianski filter),
   - izenačevanje histograma (CLAHE) za odpornost na osvetlitev.

2. **LBP histogram** (`app/lbp.py`)
   - za vsak piksel 8-bitna koda iz primerjave z 8 sosedi,
   - slika razdeljena na mrežo 8×8 celic,
   - 256-koš histogram na celico, L1-normaliziran,
   - združen vektor dolžine 8·8·256 = 16384.

3. **ORB ujemanje** (`app/orb_matching.py`)
   - ORB ključne točke + binarni deskriptorji,
   - BFMatcher (Hamming) + Lowejev ratio test,
   - podobnost = delež dobrih ujemanj.

4. **Združitev in odločitev** (`app/face_comparison.py`)
   - `score = 0.6·LBP_sim + 0.4·ORB_sim`,
   - `match = score ≥ prag`.

## Hiperparametri

| Hiperparameter | Privzeto | Opis |
|---|---|---|
| `lbp_grid` | 8×8 | finost prostorske mreže LBP |
| `lbp_weight` / `orb_weight` | 0.6 / 0.4 | uteži obeh signalov |
| `match_threshold` | 0.5 | prag odločitve |
| `orb_features` | 500 | največ ORB ključnih točk |
| `orb_ratio` | 0.75 | Lowejev ratio test |

Vse je zbrano v `app/config.py`.

## Vrednotenje

`scripts/evaluate.py` zgradi pozitivne/negativne pare, oceni vsak par in
preišče prage od 0 do 1 ter izbere tistega z najvišjim **F1** (optimizacija
hiperparametra `match_threshold`). Metrike: točnost, preciznost, priklic, F1
in matrika zmot. Delitev podatkov je po osebah (`app/dataset.py`), da ista
identiteta ne nastopa hkrati v učni in testni množici.
