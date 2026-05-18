#!/bin/bash
set -e # exit immediately if anything fails

echo "Starting ORV API"

uvicorn api.api:app --host 0.0.0.0 --port 3002