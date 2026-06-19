import os 
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace, AmlCompute
from azure.identity import ClientSecretCredential
from azure.mgmt.resource import ResourceManagementClient

load_dotenv()

credential = ClientSecretCredential(
    tenant_id= os.environ['TENANT_ID'],
    client_id= os.environ['AZURE_SERVICE_PRINCIPAL_APPID'],
    client_secret= os.environ['AZURE_SERVICE_PRINCIPAL_PASSWORD'],
)

client = MLClient(
    credential= credential,
    subscription_id= os.environ['SUBSCRIPTION_ID'],
    resource_group_name= os.environ['RESOURCE_GROUP'],
)

print("Creating Workspace...")

ws = Workspace(
    name= "MachineLearningOps",
    location="uaenorth",
)

ws = client.workspaces.begin_create(ws).result()

print(f"Workspace Created: {ws.name}")

print("Creating Compute Cluster...")

cluster = AmlCompute(
    name= 'cpu-cluster',
    type= 'amlcompute',
    size= 'STANDARD_DS3_V2',
    min_instances= 0,
    max_instances= 3,
    idle_time_before_scale_down= 180,
)

ml_client = MLClient(
    credential= credential,
    subscription_id= os.environ['SUBSCRIPTION_ID'],
    resource_group_name= os.environ['RESOURCE_GROUP'],
    workspace_name= 'MachineLearningOps'
)

ml_client.compute.begin_create_or_update(cluster).result()
print("compute cluster created: cpu-cluster")
print("ALL DONE!")


print("\nRemember to delete workspace when done to avoid costs:")
print("az ml workspace delete --name MachineLearning --resource-group mlops-rg --yes")