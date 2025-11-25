from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, dsl, load_component

# --------------------------------------------------------------------
# CONFIG – update IDs if needed
# --------------------------------------------------------------------
SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60300832"
WORKSPACE_NAME = "GoodReadsReviewsAnalysis60300832"

COMPUTE_NAME = "cpu-cluster"

# Data asset that contains your features parquet (from Silver layer)
FEATURES_DATA_ASSET = "azureml:azureml_olden_tooth_qkdfkgrpr3_output_data_output_parquet:1"

# Feature set version used only for metadata in training
FEATURE_SET_VERSION = "4"


def get_ml_client() -> MLClient:
    """Create an MLClient for the workspace."""
    cred = DefaultAzureCredential()
    return MLClient(
        credential=cred,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


# --------------------------------------------------------------------
# Load components from YAML (relative to repo root when you run python src/pipeline_job.py)
# --------------------------------------------------------------------
feature_retrieval_component = load_component(source="components/feature_retrieval.yml")
feature_selection_component = load_component(source="components/feature_selection.yml")
train_eval_component = load_component(source="components/train_eval.yml")


# --------------------------------------------------------------------
# Define the pipeline
# --------------------------------------------------------------------
@dsl.pipeline(
    compute=COMPUTE_NAME,
    description="Tumor MRI feature pipeline: retrieval -> feature selection -> training",
)
def tumor_pipeline_job(features_parquet: Input(type="uri_file")):
    # -------------------------------
    # Component A: Feature Retrieval
    # -------------------------------
    # feature_retrieval.yml v2:
    #   inputs:
    #       features_parquet (uri_file)
    #   outputs:
    #       train_parquet (uri_file)
    #       test_parquet  (uri_file)
    retrieval_step = feature_retrieval_component(
        features_parquet=features_parquet
    )

    # -------------------------------
    # Component B: Feature Selection
    # -------------------------------
    # feature_selection.yml v2:
    #   inputs:
    #       train_parquet (uri_file)
    #   outputs:
    #       selected_features_json (uri_file)
    #       baseline_metrics_json  (uri_file)
    #       ga_metrics_json        (uri_file)
    feature_selection_step = feature_selection_component(
        train_parquet=retrieval_step.outputs.train_parquet
    )

    # -------------------------------
    # Component C: Training & Eval
    # -------------------------------
    # train_eval.yml v2:
    #   inputs:
    #       train_parquet         (uri_file)
    #       test_parquet          (uri_file)
    #       selected_features_json(uri_file)
    #       feature_set_version   (string)
    #   outputs:
    #       output_dir (uri_folder, contains metrics.json + model.joblib)
    train_eval_step = train_eval_component(
        train_parquet=retrieval_step.outputs.train_parquet,
        test_parquet=retrieval_step.outputs.test_parquet,
        selected_features_json=feature_selection_step.outputs.selected_features_json,
        feature_set_version=FEATURE_SET_VERSION,
    )

    # Ensure all steps run on the desired compute cluster (optional but nice)
    retrieval_step.compute = COMPUTE_NAME
    feature_selection_step.compute = COMPUTE_NAME
    train_eval_step.compute = COMPUTE_NAME

    # Expose key artifacts as pipeline outputs
    return {
        # From A
        "train_parquet": retrieval_step.outputs.train_parquet,
        "test_parquet": retrieval_step.outputs.test_parquet,
        # From B
        "selected_features_json": feature_selection_step.outputs.selected_features_json,
        "baseline_metrics_json": feature_selection_step.outputs.baseline_metrics_json,
        "ga_metrics_json": feature_selection_step.outputs.ga_metrics_json,
        # From C
        "training_outputs": train_eval_step.outputs.output_dir,
    }


# --------------------------------------------------------------------
# Submit the pipeline
# --------------------------------------------------------------------
if __name__ == "__main__":
    ml_client = get_ml_client()

    # Input: the features parquet (Silver output) as a data asset
    features_input = Input(
        type="uri_file",
        path=FEATURES_DATA_ASSET,
    )

    pipeline_job = tumor_pipeline_job(
        features_parquet=features_input
    )

    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="tumor_mri_pipeline",
    )

    print("Pipeline submitted!")
    print("Job name:", submitted_job.name)
    if hasattr(submitted_job, "studio_url"):
        print("Studio URL:", submitted_job.studio_url)
