# SentimentOps — Command Reference

Complete reference for every command in the project.
Last updated: Phase 8

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [SentimentOps CLI](#2-sentimentops-cli)
3. [Python Scripts (Direct)](#3-python-scripts-direct)
4. [MLflow](#4-mlflow)
5. [FastAPI / uvicorn](#5-fastapi--uvicorn)
6. [Docker](#6-docker)
7. [Azure ML](#7-azure-ml)
8. [Git](#8-git)
9. [Pytest](#9-pytest)
10. [PowerShell API Testing](#10-powershell-api-testing)

---

## 1. Environment Setup

```bash
# create virtual environment
python -m venv venv

# activate (Windows PowerShell)
venv\Scripts\activate

# activate (Mac/Linux)
source venv/bin/activate

# install all dependencies
pip install -r requirements.txt

# install project as editable package (required for sentimentops CLI)
pip install -e .

# check installed packages
pip list

# check specific package version
pip show mlflow
pip show scikit-learn
```

---

## 2. SentimentOps CLI

> All commands require `pip install -e .` to be run first from project root.

### Help

```bash
# show all available commands
sentimentops --help

# show version
sentimentops --version

# show help for any specific command
sentimentops predict --help
sentimentops train-model --help
sentimentops evaluate-model --help
sentimentops cloud-train --help
```

---

### preprocess-data

Downloads IMDB dataset from HuggingFace and saves to `data/`

```bash
# download and preprocess IMDB dataset
sentimentops preprocess-data
```

---

### train-model

Trains TF-IDF + Logistic Regression model locally

```bash
# train with default params (MLflow tracking enabled)
sentimentops train-model

# train with custom params
sentimentops train-model --max-features 100000 --c 5.0

# train with custom ngram range
sentimentops train-model --max-features 75000 --ngram-range 1,3 --c 2.0

# train WITHOUT MLflow tracking
sentimentops train-model --no-track

# train WITH MLflow tracking (default)
sentimentops train-model --track

# all options with custom values
sentimentops train-model --max-features 50000 --ngram-range 1,2 --c 1.0 --track
```

| Option | Default | Description |
|---|---|---|
| `--max-features` | 50000 | TF-IDF vocabulary size |
| `--ngram-range` | 1,2 | TF-IDF ngram range |
| `--c` | 1.0 | Logistic Regression regularization |
| `--track/--no-track` | track | Enable/disable MLflow tracking |

---

### predict

Predict sentiment of a single movie review

```bash
# predict using local model (default)
sentimentops predict "This movie was absolutely fantastic"

# predict negative review
sentimentops predict "Terrible waste of time, worst movie ever"

# predict using azure-downloaded model
sentimentops predict "This movie was great" --source azure

# predict using local model explicitly
sentimentops predict "This movie was great" --source local

# predict with custom model path
sentimentops predict "great movie" --model-path data/model.joblib

# predict using MLflow run
sentimentops predict "great movie" --model-path "runs:/YOUR_RUN_ID/model"

# predict using MLflow registry
sentimentops predict "great movie" --model-path "models:/sentimentops-tfidf/latest"
```

| Option | Default | Description |
|---|---|---|
| `--source` | local | Which model to use: local or azure |
| `--model-path` | data/model.joblib | Path or MLflow URI to model |

---

### batch-predict

Predict sentiment for multiple reviews via stdin

```bash
# Windows PowerShell — pipe multiple reviews
@"
This movie was absolutely fantastic
Terrible waste of time
No Country for Old Men is a masterpiece
The acting was wooden and the plot made no sense
"@ | sentimentops batch-predict

# pipe from a text file (one review per line)
Get-Content reviews.txt | sentimentops batch-predict

# with custom model source
Get-Content reviews.txt | sentimentops batch-predict --source azure
```

---

### evaluate-model

Evaluate model performance on the test set

```bash
# evaluate with defaults
sentimentops evaluate-model

# evaluate with custom paths
sentimentops evaluate-model --model-path data/model.joblib --test-path data/test.csv

# evaluate azure model
sentimentops evaluate-model --model-path data/azure_outputs/model.joblib
```

| Option | Default | Description |
|---|---|---|
| `--model-path` | data/model.joblib | Path to model file |
| `--test-path` | data/test.csv | Path to test CSV |

---

### cloud-train

Submit a training job to Azure ML

```bash
# submit with default params
sentimentops cloud-train --max-features 50000 --c 1.0

# submit with custom params
sentimentops cloud-train --max-features 100000 --c 5.0 --ngram-range 1,3
```

| Option | Default | Description |
|---|---|---|
| `--max-features` | 50000 | TF-IDF vocabulary size |
| `--ngram-range` | 1,2 | ngram range |
| `--c` | 1.0 | Logistic Regression C |

---

## 3. Python Scripts (Direct)

> Run all scripts from the **project root** directory.

```bash
# preprocess data
python src/preprocess.py

# train model
python src/train.py

# train with custom params
python src/train.py --max_features 100000 --ngram_range 1,2 --C 5.0

# submit Azure ML job
python azure/submit_job.py

# submit Azure ML job with custom params
python azure/submit_job.py --max_features 50000 --ngram_range 1,2 --C 1.0

# create Azure workspace and cluster
python azure/create_workspace.py

# delete Azure workspace (saves costs)
python azure/delete_workspace.py
```

---

## 4. MLflow

### Local MLflow UI

```bash
# start MLflow UI (local tracking server)
mlflow ui

# start on custom port
mlflow ui --port 5001

# open in browser
# http://127.0.0.1:5000
```

### List Experiments and Runs (Python)

```bash
# list all experiments and runs
python -c "
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiments = client.search_experiments()
for exp in experiments:
    print(f'Experiment: {exp.name}')
    runs = client.search_runs(exp.experiment_id)
    for run in runs:
        print(f'  Run: {run.info.run_id}')
        print(f'  Accuracy: {run.data.metrics.get(\"accuracy\", \"N/A\")}')
        print(f'  Params: {run.data.params}')
"
```

### List Registered Models (Python)

```bash
python -c "
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
models = client.search_registered_models()
for m in models:
    print(f'Model: {m.name}')
    for v in m.latest_versions:
        print(f'  Version {v.version} | run: {v.run_id}')
"
```

### Load Model from MLflow (Python)

```bash
# load from run ID
python -c "
import mlflow.sklearn
model = mlflow.sklearn.load_model('runs:/YOUR_RUN_ID/model')
print(model.predict(['great movie']))
"

# load from registry
python -c "
import mlflow.sklearn
model = mlflow.sklearn.load_model('models:/sentimentops-tfidf/latest')
print(model.predict(['great movie']))
"
```

### Connect MLflow to Azure (Python)

```bash
python -c "
from dotenv import load_dotenv
import os, mlflow
load_dotenv()
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential

credential = ClientSecretCredential(
    tenant_id     = os.environ['TENANT_ID'],
    client_id     = os.environ['AZURE_SERVICE_PRINCIPAL_APPID'],
    client_secret = os.environ['AZURE_SERVICE_PRINCIPAL_PASSWORD'],
)
client = MLClient(
    credential          = credential,
    subscription_id     = os.environ['SUBSCRIPTION_ID'],
    resource_group_name = os.environ['RESOURCE_GROUP'],
    workspace_name      = os.environ['WORKSPACE_NAME'],
)
uri = client.workspaces.get(client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(uri)
print('Connected to Azure MLflow:', uri)
"
```

---

## 5. FastAPI / uvicorn

```bash
# start API server (from project root)
uvicorn api.main:app --reload

# start on custom port
uvicorn api.main:app --reload --port 8001

# start with custom model path
MODEL_PATH=data/azure_outputs/model.joblib uvicorn api.main:app --reload

# start with custom model version label
MODEL_VERSION=tfidf-logreg-v2 uvicorn api.main:app --reload

# start with MLflow registry model
MODEL_PATH="models:/sentimentops-tfidf/latest" uvicorn api.main:app --reload

# production mode (no reload, multiple workers)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

```
GET  http://localhost:8000/           → root info
GET  http://localhost:8000/health     → health check
GET  http://localhost:8000/docs       → Swagger UI (interactive docs)
POST http://localhost:8000/predict    → single prediction
POST http://localhost:8000/predict/batch → batch predictions
```

### Test API (Python)

```bash
python notebooks/test_docker.py
```

---

## 6. Docker

### Build

```bash
# build image
docker build -t sentimentops:latest .

# build with specific tag
docker build -t sentimentops:v1.0 .

# build with no cache (force full rebuild)
docker build --no-cache -t sentimentops:latest .

# build and see verbose output
docker build --progress=plain -t sentimentops:latest .
```

### Run

```bash
# run container
docker run -p 8000:8000 sentimentops:latest

# run in background (detached)
docker run -d -p 8000:8000 sentimentops:latest

# run with custom model version
docker run -d -p 8000:8000 -e MODEL_VERSION="tfidf-logreg-v1" sentimentops:latest

# run with local data folder mounted (model updates without rebuild)
docker run -d -p 8000:8000 -v ${PWD}/data:/app/data sentimentops:latest

# run on different host port
docker run -d -p 9000:8000 sentimentops:latest
# API now at http://localhost:9000
```

### Manage Containers

```bash
# list running containers
docker ps

# list all containers including stopped
docker ps -a

# stop a container
docker stop <container_id>

# stop all running containers
docker stop $(docker ps -q)

# remove a container
docker rm <container_id>

# remove all stopped containers
docker container prune

# view container logs
docker logs <container_id>

# follow logs in real time
docker logs -f <container_id>

# open shell inside running container
docker exec -it <container_id> /bin/bash
```

### Manage Images

```bash
# list all images
docker images

# remove an image
docker rmi sentimentops:latest

# remove all unused images
docker image prune

# check image size
docker images sentimentops
```

### Docker Compose

```bash
# start services
docker-compose up

# start in background
docker-compose up -d

# rebuild and start
docker-compose up --build

# stop services
docker-compose down

# stop and remove volumes
docker-compose down -v

# view logs
docker-compose logs

# follow logs
docker-compose logs -f
```

---

## 7. Azure ML

### Workspace Management

```bash
# create workspace and cluster
python azure/create_workspace.py

# delete workspace (saves costs — run when done)
python azure/delete_workspace.py

# check workspace exists via Azure CLI
az ml workspace show --name MachineLearningOps --resource-group mlops-rg

# restore soft-deleted workspace
az ml workspace restore --name MachineLearningOps --resource-group mlops-rg

# purge soft-deleted workspace
az ml workspace purge --name MachineLearningOps --resource-group mlops-rg --yes
```

### Job Management

```bash
# submit job (via CLI)
sentimentops cloud-train --max-features 50000 --c 1.0

# submit job (via Python script directly)
python azure/submit_job.py --max_features 50000 --C 1.0

# list recent jobs (Python)
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential

credential = ClientSecretCredential(
    tenant_id     = os.environ['TENANT_ID'],
    client_id     = os.environ['AZURE_SERVICE_PRINCIPAL_APPID'],
    client_secret = os.environ['AZURE_SERVICE_PRINCIPAL_PASSWORD'],
)
client = MLClient(
    credential          = credential,
    subscription_id     = os.environ['SUBSCRIPTION_ID'],
    resource_group_name = os.environ['RESOURCE_GROUP'],
    workspace_name      = os.environ['WORKSPACE_NAME'],
)
jobs = list(client.jobs.list())[:5]
for j in jobs:
    print(f'{j.name} | {j.status} | {j.creation_context.created_at}')
"

# cancel all queued/running jobs (Python)
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential

credential = ClientSecretCredential(
    tenant_id     = os.environ['TENANT_ID'],
    client_id     = os.environ['AZURE_SERVICE_PRINCIPAL_APPID'],
    client_secret = os.environ['AZURE_SERVICE_PRINCIPAL_PASSWORD'],
)
client = MLClient(
    credential          = credential,
    subscription_id     = os.environ['SUBSCRIPTION_ID'],
    resource_group_name = os.environ['RESOURCE_GROUP'],
    workspace_name      = os.environ['WORKSPACE_NAME'],
)
for job in client.jobs.list():
    if job.status in ['Queued', 'Starting', 'Running']:
        client.jobs.begin_cancel(job.name)
        print(f'Cancelled: {job.name}')
"

# download job outputs (Python)
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential

credential = ClientSecretCredential(
    tenant_id     = os.environ['TENANT_ID'],
    client_id     = os.environ['AZURE_SERVICE_PRINCIPAL_APPID'],
    client_secret = os.environ['AZURE_SERVICE_PRINCIPAL_PASSWORD'],
)
client = MLClient(
    credential          = credential,
    subscription_id     = os.environ['SUBSCRIPTION_ID'],
    resource_group_name = os.environ['RESOURCE_GROUP'],
    workspace_name      = os.environ['WORKSPACE_NAME'],
)
jobs = list(client.jobs.list())
latest = next(j for j in jobs if j.status == 'Completed')
client.jobs.download(
    name          = latest.name,
    output_name   = 'default',
    download_path = 'data/azure_outputs'
)
print('Downloaded to data/azure_outputs/')
"
```

### Environment Management

```bash
# list all environments in workspace (Python)
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential

credential = ClientSecretCredential(
    tenant_id     = os.environ['TENANT_ID'],
    client_id     = os.environ['AZURE_SERVICE_PRINCIPAL_APPID'],
    client_secret = os.environ['AZURE_SERVICE_PRINCIPAL_PASSWORD'],
)
client = MLClient(
    credential          = credential,
    subscription_id     = os.environ['SUBSCRIPTION_ID'],
    resource_group_name = os.environ['RESOURCE_GROUP'],
    workspace_name      = os.environ['WORKSPACE_NAME'],
)
for e in client.environments.list():
    print(e.name)
"

# archive old environments to force fresh build (Python)
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential

credential = ClientSecretCredential(
    tenant_id     = os.environ['TENANT_ID'],
    client_id     = os.environ['AZURE_SERVICE_PRINCIPAL_APPID'],
    client_secret = os.environ['AZURE_SERVICE_PRINCIPAL_PASSWORD'],
)
client = MLClient(
    credential          = credential,
    subscription_id     = os.environ['SUBSCRIPTION_ID'],
    resource_group_name = os.environ['RESOURCE_GROUP'],
    workspace_name      = os.environ['WORKSPACE_NAME'],
)
for name in ['sentimentops-env', 'sentimentops-env-v2']:
    for v in client.environments.list(name=name):
        client.environments.archive(name=name, version=v.version)
        print(f'Archived: {name} version {v.version}')
"
```

### Azure CLI

```bash
# login to Azure
az login

# set subscription
az account set --subscription YOUR_SUBSCRIPTION_ID

# list resource groups
az group list --output table

# list ML workspaces
az ml workspace list --resource-group mlops-rg

# create compute cluster via CLI
az ml compute create \
  --name cpu-cluster \
  --type amlcompute \
  --min-instances 0 \
  --max-instances 3 \
  --size STANDARD_DS3_V2 \
  --resource-group mlops-rg \
  --workspace-name MachineLearningOps
```

---

## 8. Git

```bash
# initialize repo
git init

# check status
git status

# add all files
git add .

# add specific file
git add src/train.py

# commit
git commit -m "your message"

# view commit history
git log --oneline

# create branch
git checkout -b feature/phase-9

# switch branch
git checkout main

# push to GitHub
git push origin main

# check what's being ignored
git check-ignore -v filename
```

---

## 9. Pytest

```bash
# run all tests
pytest tests/ -v

# run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# run specific test file
pytest tests/test_api.py -v
pytest tests/test_cli.py -v
pytest tests/test_preprocess.py -v
pytest tests/test_train.py -v
pytest tests/test_predict.py -v

# run specific test function
pytest tests/test_preprocess.py::test_clean_text_lowercase -v

# run and stop at first failure
pytest tests/ -v -x

# clear pytest cache
rmdir /s /q .pytest_cache
rmdir /s /q src\__pycache__
rmdir /s /q tests\__pycache__
```

---
## 10. PowerShell API Testing
# deployment commands
az acr create --resource-group mlops-rg --name sentimentopsacr --sku Basic
az acr login --name sentimentopsacr
az acr update --name sentimentopsacr --admin-enabled true
az containerapp env create --name sentimentops-env --resource-group mlops-rg --location uaenorth
az containerapp create --name sentimentops-api ...

# get live URL
az containerapp show --name sentimentops-api --resource-group mlops-rg --query properties.configuration.ingress.fqdn --output tsv

# teardown
az containerapp delete --name sentimentops-api --resource-group mlops-rg --yes
az acr delete --name sentimentopsacr --resource-group mlops-rg --yes

## 11. PowerShell API Testing

> Use these instead of curl — curl doesn't work properly in PowerShell.

```powershell
# health check
Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" | Select-Object -Expand Content

# single prediction
Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text": "This movie was absolutely fantastic"}' `
  | Select-Object -Expand Content

# negative review
Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text": "Terrible waste of time, worst movie ever"}' `
  | Select-Object -Expand Content

# batch prediction
Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict/batch" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"texts": ["Absolutely brilliant film", "Complete waste of time", "Masterpiece"]}' `
  | Select-Object -Expand Content
```

---

## Quick Reference Card

```
TASK                              COMMAND
────────────────────────────────  ──────────────────────────────────────────
Activate venv                     venv\Scripts\activate
Install dependencies              pip install -r requirements.txt
Register CLI                      pip install -e .

Preprocess data                   sentimentops preprocess-data
Train model locally               sentimentops train-model
Train with custom params          sentimentops train-model --max-features 100000 --c 5.0
Predict single review             sentimentops predict "your review here"
Evaluate model                    sentimentops evaluate-model
Submit to Azure                   sentimentops cloud-train --max-features 50000 --c 1.0

Start MLflow UI                   mlflow ui
Start API locally                 uvicorn api.main:app --reload

Build Docker image                docker build -t sentimentops:latest .
Run Docker container              docker run -p 8000:8000 sentimentops:latest
Run in background                 docker run -d -p 8000:8000 sentimentops:latest
Stop container                    docker stop <container_id>
View container logs               docker logs <container_id>

Run all tests                     pytest tests/ -v
Run tests with coverage           pytest tests/ -v --cov=src --cov-report=term-missing

Create Azure workspace            python azure/create_workspace.py
Delete Azure workspace            python azure/delete_workspace.py
Cancel all Azure jobs             (see Azure ML section above)

API docs (browser)                http://localhost:8000/docs
MLflow UI (browser)               http://localhost:5000
Azure ML Studio                   https://ml.azure.com
```

---

## Environment Variables (.env)

```env
SUBSCRIPTION_ID=your_azure_subscription_id
TENANT_ID=your_azure_tenant_id
AZURE_SERVICE_PRINCIPAL_APPID=your_service_principal_id
AZURE_SERVICE_PRINCIPAL_PASSWORD=your_service_principal_password
RESOURCE_GROUP=mlops-rg
WORKSPACE_NAME=MachineLearningOps

# optional — override defaults
MODEL_PATH=data/model.joblib
MODEL_VERSION=tfidf-logreg-v1
```

---

*SentimentOps — Phase 1 through 8 complete*
