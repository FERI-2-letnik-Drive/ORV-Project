## Install requirements

---

pip install -r requirements.txt

## Save requirements

---
**Save exact versions of all Python packages installes in this env.**

pip freeze > requirements.txt 

## Start server

---
uvicorn api.api:app --reload 
-- reload automatically refreshes on changes to file. 
