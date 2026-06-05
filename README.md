# ORV – Verifikacija obraza (2FA)

Sistem za dodatno preverjanje identitete uporabnika z metodami
računalniškega vida. Ob prijavi uporabnik s sliko obraza potrdi identiteto;
strežnik primerja sliko z referenčno in vrne odločitev (ujemanje / zavrnitev).

## Kako deluje

Primerjava obrazov združuje dva klasična postopka računalniškega vida:

- **LBP histogram** (Local Binary Patterns) – opis globalne teksture obraza,
- **ORB ujemanje značilk** – ujemanje lokalnih ključnih točk.

Rezultat je utežena vsota obeh signalov; odločitev je primerjava s pragom.
Podrobnosti v [`docs/MODEL.md`](docs/MODEL.md).

## Zagon

```bash
./start.sh
```

Strežnik teče na `http://localhost:3002`. Glej tudi:

- API: [`docs/API.md`](docs/API.md)
- Razvojno okolje: [`docs/dev-setup.md`](docs/dev-setup.md)
- Docker: [`docs/docker-setup.md`](docs/docker-setup.md)

## Zajem podatkov in vrednotenje

```bash
# zajem učnih slik z webkamere
python -m scripts.capture_faces --person ime_osebe --count 20

# vrednotenje modela in optimizacija praga
python -m scripts.evaluate --dataset data/dataset
```

## Struktura

```
api/    FastAPI strežnik (/health, /api/v1/face-verifications)
app/    računalniški vid: predobdelava, LBP, ORB, primerjava, metrike
scripts/ zajem podatkov in vrednotenje
tests/  enotni testi
docs/   dokumentacija
```

## Testi

```bash
python -m pytest
```
