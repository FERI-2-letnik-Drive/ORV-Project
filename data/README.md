# Podatki

Pričakovana struktura za zajem in vrednotenje:

```
data/dataset/
    oseba1/  oseba1_000.jpg, oseba1_001.jpg, ...
    oseba2/  oseba2_000.jpg, ...
```

Slike zajamemo z `python -m scripts.capture_faces --person oseba1`.
Mape z dejanskimi slikami **niso** verzionirane (glej `.gitignore`).
