import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main(args):
    print(f"[Feature Retrieval] Reading features from: {args.features_parquet}")
    df = pd.read_parquet(args.features_parquet)

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the input parquet.")

    # Drop rows with missing label (safety)
    df = df.dropna(subset=["label"])
    print(f"[Feature Retrieval] Rows after dropping missing labels: {len(df)}")

    # Stratified split 80/20
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    # Save outputs directly to the given file paths
    print(f"[Feature Retrieval] Writing train to: {args.train_parquet}")
    train_df.to_parquet(args.train_parquet, index=False)

    print(f"[Feature Retrieval] Writing test to: {args.test_parquet}")
    test_df.to_parquet(args.test_parquet, index=False)

    print("[Feature Retrieval] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features_parquet",
        type=str,
        required=True,
        help="Path to the full features parquet (Silver output).",
    )
    parser.add_argument(
        "--train_parquet",
        type=str,
        required=True,
        help="Output path for train.parquet",
    )
    parser.add_argument(
        "--test_parquet",
        type=str,
        required=True,
        help="Output path for test.parquet",
    )

    args = parser.parse_args()
    main(args)
