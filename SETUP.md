# ONE COMMAND SETUP

---
## Start Docker Container
```bash
docker compose up --build
```

## Clean Rebuild And Start Docker Container
```bash
docker compose build --no-cache
docker compose up
```

---
## Stop And Remove Container
```bash
docker compose down
```
---
# MANUAL SETUP

## Build Docker Image

```bash
docker build --no-cache -t orv-api:latest .
```

---

## Run Docker Container

```bash
docker run --rm --name orv_api_container -p 3002:3002 orv-api:latest
```

---

## Stop Container

Press:
CTRL + C

**AND**

```bash
docker stop orv_api_container
```
