# IDE SETUP

---
## Install requirements


```bash
pip install -r requirements-dev.txt
```
---
## Save requirements

Check installed package versions:

```bash
pip freeze
```

Update manually:

- `requirements-base.txt` → shared packages
- `requirements-dev.txt` → `opencv-python`
- `requirements-docker.txt` → `opencv-python-headless`

**Keep OpenCV versions the same in both requirement files.**

---
## Start server
```bash
uvicorn api.api:app --reload --host 127.0.0.1 --port 3002
```

**--reload flag** automatically refreshes on changes to file.