import argparse
import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(train_path, test_path, selected_features_path):
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("Expected 'label' column in train/test parquet.")

    with open(selected_features_path, "r") as f:
        sel = json.load(f)
    selected_features = sel.get("selected_features", sel)

    selected_features = [
        f for f in selected_features
        if f in train_df.columns and f not in ["label", "image_id"]
    ]

    if len(selected_features) == 0:
        raise ValueError("No valid selected features found in selected_features.json")

    X_train = train_df[selected_features].values
    y_train = train_df["label"].values

    X_test = test_df[selected_features].values
    y_test = test_df["label"].values

    return X_train, y_train, X_test, y_test, selected_features


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Train/Eval] Loading data from {args.train_parquet} and {args.test_parquet}")
    X_train, y_train, X_test, y_test, selected_features = load_data(
        args.train_parquet, args.test_parquet, args.selected_features_json
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    print("[Train/Eval] Training RandomForest...")
    clf.fit(X_train, y_train)

    print("[Train/Eval] Evaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "num_selected_features": int(len(selected_features)),
        "selected_features": selected_features,
        "feature_set_version": args.feature_set_version,
        "training_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "model_type": "RandomForestClassifier",
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Train/Eval] Wrote metrics to {metrics_path}")

    model_path = os.path.join(args.output_dir, "model.joblib")
    joblib.dump(
        {
            "model": clf,
            "selected_features": selected_features,
            "feature_set_version": args.feature_set_version,
        },
        model_path,
    )
    print(f"[Train/Eval] Wrote model to {model_path}")
    print("[Train/Eval] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_parquet",
        type=str,
        required=True,
        help="Path to train.parquet",
    )
    parser.add_argument(
        "--test_parquet",
        type=str,
        required=True,
        help="Path to test.parquet",
    )
    parser.add_argument(
        "--selected_features_json",
        type=str,
        required=True,
        help="Path to selected_features.json from Component B",
    )
    parser.add_argument(
        "--feature_set_version",
        type=str,
        default="4",
        help="Feature set version used for training (for metadata).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write metrics.json and model artifact.",
    )

    args = parser.parse_args()
    main(args)
