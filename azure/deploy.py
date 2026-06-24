import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
RESOURCE_GROUP    = os.environ["RESOURCE_GROUP"]
LOCATION          = "uaenorth"
ACR_NAME          = "sentimentopsacr"          # must be globally unique, lowercase
APP_NAME          = "sentimentops-api"
IMAGE_NAME        = f"{ACR_NAME}.azurecr.io/sentimentops:latest"


def run(command: str) -> str:
    """Run a shell command and return output."""
    print(f"\n→ {command}")
    result = subprocess.run(
        command,
        shell    = True,
        text     = True,
        encoding = "utf-8",
        errors   = "replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")
    return ""


def run_capture(command: str) -> str:
    """Run a command and capture output (for commands that return values)."""
    print(f"\n→ {command}")
    result = subprocess.run(
        command,
        shell          = True,
        capture_output = True,
        text           = True,
        encoding       = "utf-8",
        errors         = "replace",
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        raise RuntimeError(f"Command failed: {command}")
    print(result.stdout)
    return result.stdout.strip()

def deploy():

    # ── 1. Create Container Registry ──────────────────────────────────
    print("\n=== Step 1: Creating Azure Container Registry ===")
    run("az acr create --resource-group mlops-rg --name sentimentopsacr --sku Basic")

    # ── 2. Login to Registry ───────────────────────────────────────────
    print("\n=== Step 2: Logging into Registry ===")
    run("az acr login --name sentimentopsacr")

    # ── 3. Build Docker Image ─────────────────────────────────────────
    print("\n=== Step 3: Building Docker Image ===")
    run("docker build -t sentimentops:latest .")

    # ── 4. Tag and Push ───────────────────────────────────────────────
    print("\n=== Step 4: Tagging Image ===")
    run(f"docker tag sentimentops:latest {IMAGE_NAME}")

    print("\n=== Step 5: Pushing to Registry ===")
    run(f"docker push {IMAGE_NAME}")

    # ── 5. Enable Admin + Get Registry Credentials ────────────────────────
    print("\n=== Step 6: Enabling admin and getting credentials ===")
    run("az acr update --name sentimentopsacr --admin-enabled true")
    acr_password = run_capture(
        f"az acr credential show --name {ACR_NAME} "
        f"--query passwords[0].value --output tsv"
    )
    print("Credentials retrieved.")

    # ── 6. Enable Admin on Registry ───────────────────────────────────
    print("\n=== Step 6b: Enabling admin access ===")
    run(f"az acr update --name {ACR_NAME} --admin-enabled true")
    acr_password = run_capture(
        f"az acr credential show --name {ACR_NAME} "
        f"--query passwords[0].value --output tsv"
    )

    # ── 7. Deploy Container App ────────────────────────────────────────
    print("\n=== Step 7: Deploying Container App ===")
    run(
        f"az containerapp create "
        f"--name {APP_NAME} "
        f"--resource-group {RESOURCE_GROUP} "
        f"--image {IMAGE_NAME} "
        f"--target-port 8000 "
        f"--ingress external "
        f"--registry-server {ACR_NAME}.azurecr.io "
        f"--registry-username {ACR_NAME} "
        f"--registry-password {acr_password} "
        f"--cpu 1.0 --memory 2.0Gi "
        f"--min-replicas 0 --max-replicas 3 "
        f"--env-vars "
        f"MODEL_PATH=data/model.joblib "
        f"DISTILBERT_PATH=data/distilbert_model "
        f"MODEL_VERSION=tfidf-logreg-v1 "
        f"BERT_VERSION=distilbert-v1"
    )

    # ── 8. Get Public URL ─────────────────────────────────────────────
    print("\n=== Step 8: Getting Public URL ===")
    url = run_capture(
        f"az containerapp show "
        f"--name {APP_NAME} "
        f"--resource-group {RESOURCE_GROUP} "
        f"--query properties.configuration.ingress.fqdn "
        f"--output tsv"
    )

    print(f"\n{'='*50}")
    print(f"Deployment complete!")
    print(f"Public URL : https://{url}")
    print(f"API Docs   : https://{url}/docs")
    print(f"Health     : https://{url}/health")
    print(f"{'='*50}")


if __name__ == "__main__":
    deploy()