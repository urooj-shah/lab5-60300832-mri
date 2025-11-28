# DSAI3202 — Lab 5
## MLOps Pipeline & MRI Tumor Detection on Azure ML
#### Urooj Shah 60300832
---
### Introduction

This lab applies the Azure ML workflows from previous assignments to MRI image classification. The pipeline goes from raw images to feature extraction to feature store to feature selection to model training to deployment.

I've completed data ingestion, feature extraction, Feature Store registration, GA-based feature selection, model training, and manual registration. I also set up GitHub Actions for automation (Phase 5) but hit a perms issue ahem.

**All the stuff is on the other branch btw**

---

### README Organization
This README is organized as follows:

1. [Project Directory Structure](#1-project-directory-structure)  
2. [Lab Sections](#2-lab-sections)  
   - I. [Phase 1](#i-phase-1)  
   - II. [Phase 2](#ii-phase-2)  
   - III. [Phase 3](#iii-phase-3)
   - IV. [Phase 4](#iv-phase-4)
   - V. [Phase 5](#v-phase-5)
   - VI. [Phase 6](#v-phase-6)
3. [Challenges](#3-challenges)   
4. [Conclusion](#4-conclusion)

---
### 1) Project Directory Structure
Below is the structure of this GitHub repository:
```
LAB5-60300832-MRI/
│
├── .github/
│   └── workflows/
│       └── aml_pipeline.yml                  # GitHub Actions workflow (Phase 5 automation)
│
├── components/                                # Azure ML component YAML definitions
│   ├── extract_features_component.yml         # Silver extraction: GLCM + filters → parquet
│   ├── feature_retrieval.yml                 # Gold: Retrieve features from Feature Store
│   ├── feature_selection.yml                 # Gold: Baseline + GA feature selection
│   └── train_eval.yml                        # Gold: Training + evaluation (RandomForest)
│
├── data/
│   └── brain_tumor_dataset/                  # Raw MRI dataset (bronze layer)
│       ├── no/                               # images with NO tumor
│       └── yes/                              # images WITH tumor
│
├── environments/                             # Conda YAMLs for Azure ML jobs
│   ├── conda.yml                             # Shared base environment
│   ├── lab5_ga_env.yml                       # Env for GA component
│   └── lab5_silver_env.yml                   # Env for feature extraction component
│
├── featurestore/                             # Azure ML Feature Store assets
│   ├── entities/                             # Entity definitions (e.g., TumorImage)
│   └── featuresets/                          # Feature set YAML (tumor_features v1)
│
├── jobs/                                     # Optional job YAMLs for debugging pipeline components
│   └── run_extract_features_job.yml          # Test job to run Silver extraction standalone
│
├── scripts/                                  # Utility scripts (Phase 6)
│   ├── deploy_endpoint.py                    # Creates/updates online endpoint for inference
│   ├── materialize_features.py               # Triggers Feature Store materialization
│   └── test_endpoint.py                      # Sends images → endpoint to measure latency + accuracy
│
├── src/                                      # Actual Python source code for components
│   ├── extract_features.py                   # Silver: MRI → GLCM + filter features
│   ├── feature_retrieval.py                  # Retrieves features+labels from Feature Store
│   ├── feature_selection.py                  # Baseline + GA feature selector
│   ├── ingest_images.py                      # Bronze: Upload raw MRI images to ADLS
│   ├── pipeline_job.py                       # Full pipeline orchestrator (Phase 4)
│   ├── score.py                              # Online scoring script for real-time endpoint
│   └── train_and_evaluate.py                 # GA-based training → saves model.joblib + metrics
│
└── README.md                                 # Main lab documentation
```

### 2) Lab Sections

AHHHHHHHHHHHHHHHHHHHHHHHHHHHHH. sorry i just had to let that out. its late.

Below I explain each phase or try to scratch at my hippocampus to remember what the heck i did. I do know id that i did this whole lab in the cli and im proud of myself for taht

#### I. Phase 1
##### Bronze Layer: Data Ingestion

I created a script that uploads the MRI images (sorted into yes/ and no/ folders for tumors) to Azure Data Lake Storage Gen2. Then I registered this uploaded data as a dataset in Azure ML so it can be easily referenced in pipelines. This is the raw data ingestion step.

#### II. Phase 2
##### Silver Layer: Feature Extraction

So heere I built an Azure ML component that takes each MRI image, extracts useful numerical features, and saves them into a single output.parquet file (btw i had no idea where this bomboclat file got saved. it was a good 10 min man hunt to find this hidden in the alleyways under someones bed)
These features will later be used by the MLOps pipeline for feature selection and model trainin.

**What the component does**

- Loads all MRI tumor images from the dataset (yes/ and no/ folders).
- Converts each image to grayscale, handling RGB, RGBA, and different file types safely.
- Computes a full set of classical image-processing features, including:
  - Entropy
  - Gaussian blur
  - Sobel
  - Prewitt
  - Gabor (real + imaginary)
  - Hessian
  - GLCM texture features for 4 angles:
      - contrast, dissimilarity, homogeneity, ASM, energy, correlation

- Ensures all intermediate images are normalized correctly so feature extraction is stable and consistent.

- Uses multiprocessing so images are processed in parallel (PARALEL AND DISTRIBUTED COMP IS THAT U).

**Files I created**

- Component YAML:
`components/extract_features.yml` - this bozo defines the Azure ML component and specifies the inputs, outputs, environment, and command.

- Python implementation:
`src/extract_features.py` this is where all the feature extraction logic lives. It loads and preprocesses each image, computes all filters + GLCM features, collects everything into a row of numerical features andf then writes the final output.parquet file `output.parquet`

  - Logs:

  ``` 
  [INFO] num_images = 253
  [INFO] num_features = 112
  [INFO] extraction_time_seconds = 112.559
  [INFO] compute SKU (AZUREML_COMPUTE) = unknown
  [INFO] Saved features parquet to /mnt/azureml/cr/j/0e0f9e376ed74f209a45f194b512e4ff/cap/data-capability/wd/output_parquet/features.parquet
  ```
lawd pls let the above be right bc ill justkms

  

#### III. Phase 3
##### Feature Store Stuff

In this phase, I registered my features into an Azure ML Feature Store, so they can be retrieved later by the MLOps pipeline - lowkey this stuff is pretty sick ngl

I used the Azure CLI to create:

1) The Feature Store Entity (key = image_id):
    ```
    az ml feature-store-entity create \
      --file featurestore/entities/tumor_image_entity.yml \
      --resource-group rg-60300832 \
      --feature-store-name featurestore60300832
    ```
    can i just mention here how it look 40 business days to find the right command for this and how many variations of feature-store-entity create i tried like i had the whole documentation page open trynna get this rigth.
2) The Feature Set (points to Phase 2 parquet):
    ```
    az ml feature-set create \
      --file featurestore/featuresets/tumor_features/tumor_features_fs.yml \
      --resource-group rg-60300832 \
      --feature-store-name featurestore60300832
    ```
These registered both the Entity and the Feature Set.

- Materialization (skipped ts)

I attempted offline materialization backfill, but Azure blocked bc they got nothing better to do (permss issues). After discussion with the goat, Dr Oussama Djedidi, bro said verbatim "Wise ppl skip that ish" . I mean its just an optimization step, i just needed him to say that so i can move on. Literally every step after this went smoothly and this was just stupid but I understand we're trying to make use of the tool out there for us to use.

So Phase 4 proceeds normally without materialization.

Sir do a backflip next time you see me if you're reading this

#### IV. Phase 4
##### Building the Full Azure ML Pipeline

In this phase, I connected all my individual components  into one Azure ML pipeline. The goal was to take the MRI images, extract features, select the best ones, train a model, and produce an output model completely end-to-end.

-  Created the component YAML files
    - extract_features_component.yml — extracts GLCM + filter features from images (Silver layer)
    - feature_retrieval.yml — loads features+labels from the Feature Store (Gold input)
    - feature_selection.yml — runs the baseline and GA selector
    - train_eval.yml — trains a Random Forest and produces metrics + model

These YAMLs define inputs, outputs, compute, and environment for each stage.

- Python scripts for each component. Inside  `src/`, I implemented the actual logic:
  - extract_features.py → loads MRI images, computes GLCM/filters, outputs output.parquet
  - feature_retrieval.py → pulls the materialized feature set from the Feature Store
  - feature_selection.py → runs baseline + genetic algorithm to choose best features
  - train_and_evaluate.py → trains the model and saves model.joblib + metrics JSON


- Connected everything in pipeline_job.py. This is like the fmaily reunion at where everybody vomes together and becomes one happy family of course
Loaded the workspace, AML client, environments, components
- Created a pipeline job that executes steps in order:
  - Silver extract features
  - Retrieve features
  - Run GA feature selection
  - Train & evaluate model
  - Passed outputs of one step as inputs to the next
  - Targeted compute cluster: cpu-cluster
  - Submitted the pipeline to Azure ML

This allowed the entire workflow to run automatically without manually triggering each component.

- I tested the pipeline in Azure ML Studio
  - After running pipeline_job.py, I checked:
  - Step-by-step job outputs
  - Metrics for baseline vs GA
  - The saved model in output_dir
  - The registered Feature Store outputs

the full pipeline produced a trained Random Forest model using GA-selected features.

#### Model Metrics & Results 

Ran three experiments: baseline model, GA feature selection, and final trained model.

##### 1. Baseline Metrics — `baseline_metrics.json`
```
{
  "baseline_accuracy": 0.7073170731707317,
  "baseline_num_features": 112
}
```
**Results:** 70.7% accuracy using all 112 features. 

#### 2. GA Metrics — `ga_metrics.json`
```
{
  "ga_accuracy": 0.7073170731707317,
  "ga_num_features": 53,
  "ga_runtime_seconds": 21.8625
}
```
**Results:** Reduced features from 112 → 53  while maintaining 70.7% accuracy. Removed redundant features and improved efficiency.

#### 3. Final Training Metrics — `metrics.json`
```
{
  "accuracy": 0.8039,
  "confusion_matrix": [[13,7],[3,28]],
  "num_selected_features": 53,
  "model_type": "RandomForestClassifier"
}
```
**Results:** 80.4% accuracy using GA-selected features. 

**Confusion matrix:** 28 TP, 13 TN, 7 FP, 3 FN.

**Summary:** Baseline: 70.7% → GA selection: 70.7% (half the features) → Final model: 80.4% 


| Model Version                   | Features | Accuracy  |
| ------------------------------- | -------- | --------- |
| Baseline RF (no selection)      | 112      | **0.707** |
| GA-Estimated Fitness            | 53       | **0.707** |
| Final RF (GA-selected features) | 53       | **0.804** |


#### V. Phase 5
##### Automation with GitHub Actions

Created `.github/workflows/aml_pipeline.yml` to automate the MLOps pipeline on pushes to `lab5_mlops_tumor_detection` or manual dispatch.

**What I completed:**
- Set up Python 3.10 environment with Azure ML SDK
- Configured OIDC authentication using Service Principal `lab5-github-sp-60300832`
- Added GitHub secrets: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_CREDENTIALS`
- Created pipeline execution step (`python src/pipeline_job.py`)
- Created deployment script (`scripts/deploy_endpoint.py`) for managed online endpoint

**Where I got stuck:**
The Service Principal lacks the contributor? role or whatever on my workspace/subscription. Authentication works, but authorization fails with "No subscriptions found" and "Client does not have authorization" errors. 

So GitHub Actions can't run pipeline jobs, register models, or create deployments. 

Other than that Workflow is fully configured and ready to run once permissions are granted. But according to Professor Utonium, the only other permission he has left to give me is owner so idk.

#### VI. Phase 6

to be continued.... (absolutely not i am officially clocked out of this course, if the All Knowing Cloud Professor uploads another version or fixes perm, I will not be continuing this lab I have done enough,  im on vacation in Diligafistan right now)

### Challenges
#### AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH. thank u.

### Conclusion

This lab guided me through the full MLOps lifecycle for MRI tumor detection — from feature engineering and feature store integration to automated training pipelines and real-time deployment. I built a modular Azure ML pipeline with feature retrieval, baseline + GA feature selection, and final model training, and I successfully registered and packaged the trained model. I also set up a GitHub Actions workflow to automate retraining and deployment, with only role permissions blocking the final execution. Overall, the lab helped me understand how real-world ML systems are structured, automated, and deployed at scale using Azure’s MLOps tools. 

