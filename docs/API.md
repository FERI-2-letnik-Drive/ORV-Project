# API

Base URL:

```txt
http://localhost:3002
```

---

## Face Verification

Endpoint:

```http
POST /api/v1/face-verifications
```

### Request

Send images using **multipart/form-data.**

### Example request

<img width="1627" height="467" alt="image" src="https://github.com/user-attachments/assets/4d8fc514-8ad0-4abf-af79-becc45d32290" />

### Example response

<img width="1627" height="271" alt="image" src="https://github.com/user-attachments/assets/bbbc7088-685a-4a3d-afad-38f0d09f0937" />

---

## Polja odgovora

| Polje | Tip | Opis |
|---|---|---|
| `match` | bool | ali gre za isto osebo (score ≥ prag) |
| `confidence` | float | združeni rezultat v območju [0, 1] |
| `message` | string | razčlenitev (LBP, ORB, prag) ali razlog napake |

### Statusne kode

- `200` – uspešna primerjava (tudi če `match=false`)
- `400` – prazna ali nedekodljiva slika
- `422` – manjka katera od zahtevanih slik
