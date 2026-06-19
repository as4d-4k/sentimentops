import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import ClientSecretCredential
from azure.mgmt.resource import ResourceManagementClient
load_dotenv()

def get_ml_client():
    """
    Authenticate and return an MLClient connected
    to ML Workspace
    """
    credential = ClientSecretCredential(
        tenant_id = os.environ['TENANT_ID'],
        client_id = os.environ['AZURE_SERVICE_PRINCIPAL_APPID'],
        client_secret= os.environ['AZURE_SERVICE_PRINCIPAL_PASSWORD'],
    )

    client = MLClient(
        credential=credential,
        subscription_id=os.environ['SUBSCRIPTION_ID'],
        resource_group_name=os.environ['RESOURCE_GROUP'],
        workspace_name=os.environ['WORKSPACE_NAME'],
    )
    return client

def submit_training_job(
        max_features:int = 50000,
        ngram:str = "1,2",
        C:float = 1.0,
):
    """
    submitting a training job to azure ML with 
    the Parameters passed as CLI arguments
    """
    #1: Connecting to Workspace
    print("Connecting to Azure ML workspace...")
    client = get_ml_client()

    #2: Defin ENV (what packages to install on azure)
    env = Environment(
        name="sentimentops-env",
        image='mcr.microsoft.com/azureml/base:latest',
        conda_file='azure/conda_env.yml',
    )

    #3: Defining the Job
    job = command(
        code= ".",
        command=(
            f"python src/train.py"
            f"--max_features {max_features}"
            f"--ngram {ngram}"
            f"C {C}"
        ),
        environment= env,
        compute= 'cpu-cluster',
        display_name='sentimentops-training',
        description= "TF-IDF + LogReg sentiment training job",
        experiment_name='sentimentops-tfidf-logreg',
    )

    #4: Submit the job
    print("Submitting Job...")
    returned_job = client.jobs.create_or_update(job)

    print("\n Job submitted Successfully!")
    print(f"Job Name: {returned_job.name}")
    print(f"Job status: {returned_job.status}")
    print(f"Studio URL: {returned_job.studio_url}")

if __name__ == "__main__":
    submit_training_job()