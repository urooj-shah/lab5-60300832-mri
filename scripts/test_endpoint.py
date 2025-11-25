import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from PIL import Image
import numpy as np


SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60300832"
WORKSPACE_NAME = "GoodReadsReviewsAnalysis60300832"
ENDPOINT_NAME = "tumor-ga-endpoint-60300832"


def get_ml_client() -> MLClient:
    cred = DefaultAzureCredential()
    return MLClient(
        credential=cred,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def encode_image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def infer_label_from_path(path: Path) -> int:
    """
    Simple heuristic:
    - if parent directory contains 'tumor' -> 1
    - else -> 0

    Adjust to match your dataset organization.
    """
    parent = path.parent.name.lower()
    if "tumor" in parent:
        return 1
    return 0


def main(args):
    ml_client = get_ml_client()

    image_paths = sorted(
        [p for p in Path(args.images_dir).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not image_paths:
        print(f"No images found in {args.images_dir}")
        return

    latencies = []
    y_true = []
    y_pred = []

    for p in image_paths:
        b64 = encode_image_to_base64(p)
        payload = json.dumps({"image_base64": b64})

        t0 = time.perf_counter()
        response_bytes = ml_client.online_endpoints.invoke(
            endpoint_name=ENDPOINT_NAME,
            request_body=payload,
            deployment_name=args.deployment_name,
        )
        dt = time.perf_counter() - t0
        latencies.append(dt)

        resp = json.loads(response_bytes)
        pred_label = resp.get("prediction")
        print(p.name, "->", resp)

        # Map back to numeric for accuracy (assuming "tumor"/"no_tumor")
        if pred_label == "tumor":
            pred = 1
        else:
            pred = 0

        y_true.append(infer_label_from_path(p))
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()
    latencies_ms = np.array(latencies) * 1000.0
    p95 = float(np.percentile(latencies_ms, 95))

    print("----- Endpoint evaluation -----")
    print(f"Num samples: {len(image_paths)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Average latency (ms): {latencies_ms.mean():.2f}")
    print(f"p95 latency (ms): {p95:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Folder with test images (e.g., contains 'tumor/' and 'no_tumor/' subfolders).",
    )
    parser.add_argument(
        "--deployment_name",
        type=str,
        default="ga-rf-deployment",
        help="Name of the managed online deployment.",
    )
    args = parser.parse_args()
    main(args)
