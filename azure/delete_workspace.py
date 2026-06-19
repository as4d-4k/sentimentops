import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from azure.mgmt.resource import ResourceManagementClient

load_dotenv()

credential = ClientSecretCredential(
    tenant_id     = os.environ["TENANT_ID"],
    client_id     = os.environ["AZURE_SERVICE_PRINCIPAL_APPID"],
    client_secret = os.environ["AZURE_SERVICE_PRINCIPAL_PASSWORD"],
)

client = MLClient(
    credential          = credential,
    subscription_id     = os.environ["SUBSCRIPTION_ID"],
    resource_group_name = os.environ["RESOURCE_GROUP"],
    workspace_name      = os.environ["WORKSPACE_NAME"],
)

print("Deleting workspace and all resources...")
client.workspaces.begin_delete(
    name              = os.environ["WORKSPACE_NAME"],
    delete_dependent_resources = True,
).result()
print("Workspace deleted. No more billing.")