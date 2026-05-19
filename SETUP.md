# ONE COMMAND SETUP

---
## Build And Start Docker Container
```bash
docker compose up --build
```

## Start Docker Container Without Build
```bash
docker compose up
```

## Clean Rebuild And Start Docker Container
```bash
docker compose build --no-cache
docker compose up
```

## Stop And Remove Container
```bash
docker compose down
```

## Stop Container
```bash
docker compose stop
```
---
# MANUAL SETUP

## Build Docker Image

Normal build:
```bash
docker build -t orv-api:latest .
```
**OR**

Clean Rebuild:
```bash
docker build --no-cache -t orv-api:latest .
```

## Run Docker Container

```bash
docker run --rm --name orv_api_container -p 3002:3002 orv-api:latest
```

## Stop Container

In other terminal: 
```bash
docker stop orv_api_container
```

**OR**

**CTRL + C** and then: 
```bash
docker stop orv_api_container
```
