import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from deap import base, creator, tools, algorithms


def load_data(train_parquet_path):
    df = pd.read_parquet(train_parquet_path)

    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in train.parquet")

    feature_cols = [c for c in df.columns if c not in ["label", "image_id"]]
    X = df[feature_cols].values
    y = df["label"].values

    return X, y, feature_cols


def run_baseline_feature_selection(X, y, feature_names, random_state=42):
    """
    Baseline:
    - VarianceThreshold to drop zero-variance features
    - Train a quick RandomForest and compute accuracy
    """
    vt = VarianceThreshold(threshold=0.0)
    X_reduced = vt.fit_transform(X)

    selected_mask = vt.get_support()
    selected_features = [f for f, keep in zip(feature_names, selected_mask) if keep]

    X_train, X_val, y_train, y_val = train_test_split(
        X_reduced,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    metrics = {
        "baseline_accuracy": float(acc),
        "baseline_num_features": int(len(selected_features)),
    }

    return selected_features, metrics


def run_ga_feature_selection(X, y, feature_names, random_state=42):
    """
    Genetic Algorithm feature selection using DEAP.
    Individual = binary mask over features.
    Fitness = validation accuracy - 0.001 * num_features
    """
    np.random.seed(random_state)

    n_features = X.shape[1]
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    POP_SIZE = 20
    N_GEN = 10
    CX_PB = 0.5
    MUT_PB = 0.2

    # Guard against double creation if imported multiple times
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=n_features,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(individual):
        if sum(individual) == 0:
            return -9999.0,

        mask = np.array(individual, dtype=bool)
        X_train_sel = X_train[:, mask]
        X_val_sel = X_val[:, mask]

        clf = RandomForestClassifier(
            n_estimators=80,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X_train_sel, y_train)
        y_pred = clf.predict(X_val_sel)
        acc = accuracy_score(y_val, y_pred)

        num_feats = mask.sum()
        fitness = acc - 0.001 * num_feats
        return float(fitness),

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=POP_SIZE)
    start_time = time.time()

    pop, _ = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=CX_PB,
        mutpb=MUT_PB,
        ngen=N_GEN,
        verbose=False,
    )

    runtime = time.time() - start_time

    best_ind = tools.selBest(pop, k=1)[0]
    best_mask = np.array(best_ind, dtype=bool)
    selected_features = [f for f, keep in zip(feature_names, best_mask) if keep]

    X_train_sel = X_train[:, best_mask]
    X_val_sel = X_val[:, best_mask]
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_val_sel)
    acc = accuracy_score(y_val, y_pred)

    metrics = {
        "ga_accuracy": float(acc),
        "ga_num_features": int(best_mask.sum()),
        "ga_runtime_seconds": float(runtime),
        "selected_feature_names": selected_features,
    }

    return selected_features, metrics


def main(args):
    print(f"[Feature Selection] Loading train data from: {args.train_parquet}")
    X, y, feature_names = load_data(args.train_parquet)

    print("[Feature Selection] Running baseline feature selection...")
    baseline_features, baseline_metrics = run_baseline_feature_selection(
        X, y, feature_names
    )

    print("[Feature Selection] Running GA feature selection...")
    ga_features, ga_metrics = run_ga_feature_selection(
        X, y, feature_names
    )

    # GA-selected features used downstream
    selected_features = ga_features

    # Write outputs to their explicit file paths
    with open(args.selected_features_json, "w") as f:
        json.dump({"selected_features": selected_features}, f, indent=2)

    with open(args.baseline_metrics_json, "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    with open(args.ga_metrics_json, "w") as f:
        json.dump(ga_metrics, f, indent=2)

    print(f"[Feature Selection] Wrote selected_features to {args.selected_features_json}")
    print(f"[Feature Selection] Wrote baseline metrics to {args.baseline_metrics_json}")
    print(f"[Feature Selection] Wrote GA metrics to {args.ga_metrics_json}")
    print("[Feature Selection] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_parquet",
        type=str,
        required=True,
        help="Path to train.parquet produced by Component A",
    )
    parser.add_argument(
        "--selected_features_json",
        type=str,
        required=True,
        help="Output path for selected_features.json",
    )
    parser.add_argument(
        "--baseline_metrics_json",
        type=str,
        required=True,
        help="Output path for baseline_metrics.json",
    )
    parser.add_argument(
        "--ga_metrics_json",
        type=str,
        required=True,
        help="Output path for ga_metrics.json",
    )
    args = parser.parse_args()
    main(args)
