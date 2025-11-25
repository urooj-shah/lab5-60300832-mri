import os
import json
import base64
import io

import joblib
import numpy as np
import pandas as pd
from PIL import Image

# If you have reusable functions in extract_features.py, you can import them:
# from extract_features import compute_features_for_image  # TODO: implement in your file

model = None
selected_features = None
feature_set_version = None


def init():
    """Called once when the deployment starts."""
    global model, selected_features, feature_set_version

    # AZUREML_MODEL_DIR points to the folder with model files
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "model.joblib")

    print(f"[init] Loading model from: {model_path}")
    artifact = joblib.load(model_path)

    model = artifact["model"]
    selected_features = artifact["selected_features"]
    feature_set_version = artifact.get("feature_set_version", "unknown")

    print(f"[init] Loaded model. {len(selected_features)} selected features.")
    print(f"[init] Feature set version: {feature_set_version}")


def _compute_features_from_image(img: Image.Image) -> pd.DataFrame:
    """
    IMPORTANT:
    Replace this with the SAME logic you used in your Silver feature extraction
    (filters + GLCM, etc.), producing a single-row DataFrame where column
    names match the training features.

    For now, this is just a placeholder that flattens the image.
    """
    img_gray = img.convert("L")
    arr = np.array(img_gray, dtype=np.float32).flatten()
    data = {f"pix_{i}": [v] for i, v in enumerate(arr)}
    df = pd.DataFrame(data)
    return df


def run(raw_data):
    """
    Expected input JSON:
    {
      "image_base64": "<base64-encoded image bytes>"
    }
    """
    try:
        if isinstance(raw_data, (bytes, bytearray)):
            raw_data = raw_data.decode("utf-8")

        data = json.loads(raw_data)
        b64 = data.get("image_base64")
        if b64 is None:
            return {"error": "No 'image_base64' field in request."}

        # Decode base64 to bytes and open as image
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes))

        # Compute full feature set
        features_df = _compute_features_from_image(img)

        # Subset to GA-selected features
        # In your REAL implementation, selected_features should be present in features_df
        available_features = [f for f in selected_features if f in features_df.columns]
        if len(available_features) == 0:
            return {"error": "No selected features found in computed features."}

        X = features_df[available_features].values

        # Predict
        probs = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = model.classes_[pred_idx]

        # Map 0/1 to human-readable label if needed
        # Adjust according to your dataset
        label_map = {0: "no_tumor", 1: "tumor"}
        label_str = label_map.get(pred_label, str(pred_label))

        return {
            "prediction": label_str,
            "probabilities": {
                str(int(c)): float(p) for c, p in zip(model.classes_, probs)
            },
            "feature_set_version": feature_set_version,
        }

    except Exception as e:
        return {"error": str(e)}
