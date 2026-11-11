"""Run the churn experiment with one or more registered classifiers."""

import argparse
import json
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from churn_data import FEATURES, TARGET, load_dataset, make_preprocessor, split_dataset
from churn_evaluation import THRESHOLD, ChurnExperiment, evaluate
from churn_figures import FigureWriter
from churn_models import MODEL_FACTORIES

HERE = Path(__file__).resolve().parent
VALIDATION_PROBE_THRESHOLD = 0.30
EXAMPLE = {
    "Geography": "France", "CreditScore": 600, "Gender": "Male", "Age": 40,
    "Tenure": 3, "Balance": 60000, "NumOfProducts": 2, "HasCrCard": 1,
    "IsActiveMember": 1, "EstimatedSalary": 50000,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "Churn_Modelling.csv")
    parser.add_argument("--output", type=Path, default=HERE.parent / "results-development")
    parser.add_argument("--evaluate-test", action="store_true", help="Explicitly evaluate the historical test partition; never use it for tuning.")
    parser.add_argument("--models", nargs="+", choices=[name for name in MODEL_FACTORIES if name != "majority"],
                        default=["ann", "random-forest"], help="The majority baseline always runs.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.output / "metrics.json").exists():
        parser.error("Output already contains metrics.json; choose a new --output directory to preserve run provenance.")

    data = load_dataset(args.data)
    splits = split_dataset(data, args.seed)
    figures = FigureWriter(args.output / "figures")
    model_dir = args.output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "provenance": {"python": platform.python_version(), "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                       "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "matplotlib"]},
                       "test_evaluated": args.evaluate_test,
                       "note": "The repository's original test results were already inspected; this partition is not a fresh final holdout."},
        "seed": args.seed,
        "threshold": THRESHOLD,
        "dataset": {
            "rows": len(data), "columns": len(data.columns),
            "missing_cells": int(data.isna().sum().sum()),
            "duplicate_rows": int(data.duplicated().sum()),
            "duplicate_customer_ids": int(data["CustomerId"].duplicated().sum()) if "CustomerId" in data else None,
            "duplicate_predictor_rows": int(data.duplicated(FEATURES).sum()),
            "class_counts": data[TARGET].value_counts().sort_index().to_dict(),
        },
        "splits": {
            name: {"rows": len(rows), "class_counts": rows[TARGET].value_counts().sort_index().to_dict()}
            for name, rows in splits.items()
        },
        "metrics": {"validation": {}, "test": {}},
        "validation_threshold_probe": {"threshold": VALIDATION_PROBE_THRESHOLD, "metrics": {}},
        "example_churn_probability": {},
        "model_parameters": {},
    }
    scores_by_split = {"validation": {}}
    if args.evaluate_test:
        scores_by_split["test"] = {}
    pd.concat([pd.DataFrame({"source_row_index": rows.index, "split": name}) for name, rows in splits.items()]).sort_values("source_row_index").to_csv(args.output / "split_manifest.csv", index=False)
    figures.class_distribution(data[TARGET])
    for name in dict.fromkeys(["majority", *args.models]):
        print(f"Training {name}", flush=True)
        classifier = MODEL_FACTORIES[name](args.seed)
        results["model_parameters"][name] = classifier.get_params()
        for package in (["tensorflow", "keras"] if name == "ann" else ["xgboost"] if name == "xgboost" else []):
            results["provenance"]["versions"][package] = version(package)
        experiment = ChurnExperiment(make_preprocessor(), classifier)
        experiment.fit(splits["train"])
        validation_labels = splits["validation"][TARGET]
        validation_scores = experiment.score(splits["validation"])
        scores_by_split["validation"][name] = validation_scores
        results["metrics"]["validation"][name] = evaluate(validation_labels, validation_scores)
        if args.evaluate_test:
            scores_by_split["test"][name] = experiment.score(splits["test"])
            results["metrics"]["test"][name] = evaluate(splits["test"][TARGET], scores_by_split["test"][name])
        results["validation_threshold_probe"]["metrics"][name] = evaluate(
            validation_labels, validation_scores, VALIDATION_PROBE_THRESHOLD,
        )
        example_score = float(experiment.score(pd.DataFrame([EXAMPLE]))[0])
        results["example_churn_probability"][name] = example_score
        joblib.dump(experiment.pipeline, model_dir / f"{name}.joblib", compress=3)
        if hasattr(classifier, "training_loss_"):
            results["ann_training_loss"] = classifier.training_loss_
            figures.training_loss(classifier.training_loss_)

    for split, scores in scores_by_split.items():
        metrics = results["metrics"][split]
        figures.precision_recall(splits[split][TARGET], scores, metrics, split)
        figures.confusion_matrices(metrics, split)
        pd.DataFrame({"source_row_index": splits[split].index, "Exited": splits[split][TARGET].to_numpy(), **scores}).to_csv(args.output / f"{split}_predictions.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, metrics in results["metrics"]["validation"].items():
        print(f"{name}: validation F1={metrics['f1']:.3f}, AP={metrics['average_precision']:.3f}")
    print(f"Wrote metrics, figures and models to {args.output.resolve()}")


if __name__ == "__main__":
    main()
