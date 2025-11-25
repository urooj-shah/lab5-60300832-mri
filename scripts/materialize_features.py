from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    MaterializationSettings,
    MaterializationComputeResource,
)

# -----------------------------
# 1. CONNECT TO FEATURE STORE
# -----------------------------
subscription_id = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
resource_group = "rg-60300832"
featurestore_name = "featurestore60300832"

credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=featurestore_name,   # Feature store IS a workspace
)

# -----------------------------
# 2. LOAD EXISTING FEATURE SET
# -----------------------------
fs = ml_client.feature_sets.get(name="tumor_features", version="4")

# -----------------------------
# 3. UPDATE MATERIALIZATION SETTINGS
# -----------------------------
fs.materialization_settings = MaterializationSettings(
    offline_enabled=True,
    online_enabled=False,

    resource=MaterializationComputeResource(
        instance_type="standard_e8s_v3"
    ),

    spark_configuration={
        "spark.driver.cores": 1,
        "spark.driver.memory": "4g",
        "spark.executor.cores": 1,
        "spark.executor.memory": "4g",
        "spark.executor.instances": 2,
    },

    schedule=None,
)

# -----------------------------
# 4. APPLY UPDATE TO AZURE
# -----------------------------
poller = ml_client.feature_sets.begin_create_or_update(fs)
result = poller.result()

print("Updated materialization settings:")
print(result)
