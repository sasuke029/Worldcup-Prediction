"""
End-to-end training script.

Usage (from the project root, with the venv activated):
    python -m src.train

Reads data/raw/results.csv and data/raw/fifa_ranking.csv, builds the
feature matrix, trains the from-scratch logistic regression model,
reports held-out and cross-validated metrics, and exports the trained
weights + scaler to reports/model_weights.json so the frontend
predictor can use your exact numbers instead of illustrative ones.
"""

from io import TextIOWrapper
import json
import argparse
from pathlib import Path

import numpy as np
from pandas import DataFrame

from .data_prep import build_dataset, FEATURE_COLS
from .model import LogisticRegressionScratch
from .cross_validation import train_test_split_stratified, standard_scale, k_fold_cv
from .metrics import evaluate_all

ROOT: Path = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model for World Cup predictions."
    )
    parser.add_argument("--results", default=str(ROOT / "data/raw/results.csv"),
                        help="Path to results CSV file")
    parser.add_argument("--rankings", default=str(ROOT / "data/raw/fifa_ranking.csv"),
                        help="Path to FIFA rankings CSV file")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Learning rate for gradient descent")
    parser.add_argument("--iterations", type=int, default=10_000,
                        help="Number of training iterations")
    parser.add_argument("--l2", type=float, default=0.0,
                        help="L2 regularization lambda")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--out", default=str(ROOT / "reports/model_weights.json"),
                        help="Output path for model weights JSON")
    args: argparse.Namespace = parser.parse_args()

    # Validate input files exist
    results_path = Path(args.results)
    rankings_path = Path(args.rankings)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    if not rankings_path.exists():
        raise FileNotFoundError(f"Rankings file not found: {rankings_path}")

    try:
        print("Building dataset...")
        X, y, model_df = build_dataset(args.results, args.rankings)
        print(f"Final dataset: {X.shape[0]} rows, {X.shape[1]} features")
        print(f"Class balance (win rate): {y.mean():.4f}")

        if X.shape[0] == 0:
            raise ValueError("Dataset is empty. Check your input CSV files.")

        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y, test_size=0.2, seed=args.seed
        )
        X_train_s, X_test_s, mu, sigma = standard_scale(X_train, X_test)

        print("\nTraining on the train split...")
        model = LogisticRegressionScratch(
            learning_rate=args.learning_rate,
            n_iterations=args.iterations,
            l2_lambda=args.l2,
            verbose=True,
        ).fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)
        test_metrics = evaluate_all(y_test, y_pred, y_proba)
        print("\nHeld-out test metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  baseline (majority class): {1 - y_test.mean():.4f}")

        print("\nRunning 5-fold cross-validation...")
        cv_df: DataFrame = k_fold_cv(
            X, y, k=5, seed=args.seed,
            learning_rate=args.learning_rate,
            n_iterations=args.iterations,
            l2_lambda=args.l2,
        )
        print(cv_df)
        print("\nMean ± std across folds:")
        print(cv_df[["accuracy", "precision", "recall", "f1", "auc"]].agg(["mean", "std"]))

        print("\nRe-fitting on the FULL dataset for deployment/export...")
        X_full_s, _, mu_full, sigma_full = standard_scale(X, X)
        final_model = LogisticRegressionScratch(
            learning_rate=args.learning_rate,
            n_iterations=args.iterations,
            l2_lambda=args.l2,
        ).fit(X_full_s, y)

        export = {
            "feature_order": FEATURE_COLS,
            "weights": dict(zip(FEATURE_COLS, final_model.w.tolist())),
            "bias": final_model.b,
            "scaler_mean": dict(zip(FEATURE_COLS, mu_full.tolist())),
            "scaler_std": dict(zip(FEATURE_COLS, sigma_full.tolist())),
            "trained_on_rows": int(X.shape[0]),
            "held_out_test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "cv_auc_mean": float(cv_df["auc"].mean()),
            "cv_auc_std": float(cv_df["auc"].std()),
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(export, f, indent=2)
        print(f"\nExported trained weights + scaler to {out_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"Data Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error during training: {e}")
        exit(1)


if __name__ == "__main__":
    try:
        main()
        print("\nTraining completed successfully!")
        print("Export file is ready for the frontend predictor.")
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
