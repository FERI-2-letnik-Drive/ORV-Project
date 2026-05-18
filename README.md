# IDE SETUP

---
## Install requirements

---
```bash
pip install -r requirements.txt
```

## Save requirements

---
**Save exact versions of all Python packages installes in this env.**
```bash
pip freeze > requirements.txt 
```

## Start server

---
```bash
uvicorn api.api:app --reload --host 127.0.0.1 --port 3002
```

**--reload flag** automatically refreshes on changes to file. 
