#!/usr/bin/env python3
"""
Ingest local MRI tumor images into ADLS Gen2 (Blob Storage-compatible)
for the Bronze layer.

Local dataset structure (relative to repo root):
    data/
      brain_tumor_dataset/
        yes/
        no/

Azure Storage:
    Account: lab5tumor60300832
    Container: lakehouse
    Paths created:
        lakehouse/raw/tumor_images/yes/
        lakehouse/raw/tumor_images/no/

Environment variables required:
    AZURE_STORAGE_CONNECTION_STRING  (your storage account connection string)
    AZURE_STORAGE_CONTAINER          (set to "lakehouse")

This script is:
- Fully programmatic (no manual uploads in portal)
- Idempotent: skips blobs that already exist
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContainerClient


# ---------- Configuration helpers ----------

def get_container_client() -> ContainerClient:
    """
    Create a ContainerClient using the connection string and container name.

    Environment variables:
        AZURE_STORAGE_CONNECTION_STRING: connection string to storage account
        AZURE_STORAGE_CONTAINER: container name, e.g. "lakehouse"
    """
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER", "lakehouse")

    if not conn_str:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING is not set. "
            "Please set it in your environment."
        )

    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service_client.get_container_client(container_name)

    # Create container if it doesn't exist (safe to call repeatedly)
    try:
        container_client.create_container()
        print(f"Created container '{container_name}'.")
    except ResourceExistsError:
        print(f"Container '{container_name}' already exists. Using existing container.")

    return container_client


# ---------- Core logic ----------

def upload_image(
    container_client: ContainerClient,
    local_path: Path,
    blob_path: str
) -> bool:
    """
    Upload a single image if it does not already exist.

    Returns True if uploaded, False if skipped.
    """
    blob_client = container_client.get_blob_client(blob_path)

    if blob_client.exists():
        print(f"[SKIP] Blob already exists: {blob_path}")
        return False

    with local_path.open("rb") as f:
        blob_client.upload_blob(f)
    print(f"[UPLOAD] {local_path} -> {blob_path}")
    return True


def ingest_dataset(
    local_root: Path,
    container_client: ContainerClient,
    base_prefix: str = "raw/tumor_images"
) -> Tuple[int, int]:
    """
    Ingest images from local_root/yes and local_root/no
    into container at:
        lakehouse/raw/tumor_images/yes/
        lakehouse/raw/tumor_images/no/

    Returns (num_uploaded, num_skipped).
    """
    if not local_root.exists() or not local_root.is_dir():
        raise RuntimeError(f"Local dataset root '{local_root}' does not exist or is not a directory.")

    uploaded = 0
    skipped = 0

    for label in ["yes", "no"]:
        local_label_dir = local_root / label
        if not local_label_dir.exists():
            print(f"[WARN] Local label directory missing: {local_label_dir} (skipping)")
            continue

        # Example blob prefix: "raw/tumor_images/yes"
        blob_prefix = f"{base_prefix}/{label}"

        # Walk through files in this label directory
        for path in sorted(local_label_dir.rglob("*")):
            if not path.is_file():
                continue

            # Construct blob path: e.g. raw/tumor_images/yes/filename.png
            blob_path = f"{blob_prefix}/{path.name}"

            try:
                if upload_image(container_client, path, blob_path):
                    uploaded += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"[ERROR] Failed to upload {path} -> {blob_path}: {e}", file=sys.stderr)

    return uploaded, skipped


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest MRI tumor images into ADLS Gen2 Bronze layer."
    )
    parser.add_argument(
        "--local_dataset_root",
        type=str,
        required=True,
        help=(
            "Path to local dataset root containing 'yes/' and 'no/' subfolders. "
            "Example: ./data/brain_tumor_dataset"
        ),
    )
    parser.add_argument(
        "--base_prefix",
        type=str,
        default="raw/tumor_images",
        help=(
            "Base path prefix inside the container. "
            "Final layout: <container>/<base_prefix>/yes/ and no/ "
            "(default: 'raw/tumor_images')."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    local_root = Path(args.local_dataset_root).resolve()
    container_client = get_container_client()

    print(f"Local dataset root: {local_root}")
    print(f"Container: {container_client.container_name}")
    print(f"Base prefix: {args.base_prefix}")
    print("Starting ingestion...")

    uploaded, skipped = ingest_dataset(
        local_root=local_root,
        container_client=container_client,
        base_prefix=args.base_prefix,
    )

    print("----- Ingestion summary -----")
    print(f"Uploaded new files : {uploaded}")
    print(f"Skipped existing   : {skipped}")


if __name__ == "__main__":
    main()
