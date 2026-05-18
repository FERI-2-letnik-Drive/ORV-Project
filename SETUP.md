# SETUP

## Build Docker image

```bash
docker build --no-cache -t orv-api:latest .
```

---

## Run Docker container

```bash
docker run --rm --name orv_api_container -p 3002:3002 orv-api:latest
```

---

## Stop container

Press:
CTRL + C

**AND**

```bash
docker stop orv_api_container
```

### TODO: ADD docker-compose.yml to handle this. 