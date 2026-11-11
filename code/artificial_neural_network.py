"""Run the churn experiment with one or more registered classifiers."""

import argparse
import json
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
    parser.add_argument("--output", type=Path, default=HERE.parent / "results")
    parser.add_argument("--models", nargs="+", choices=[name for name in MODEL_FACTORIES if name != "majority"],
                        default=["ann", "random-forest"], help="The majority baseline always runs.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_dataset(args.data)
    splits = split_dataset(data, args.seed)
    figures = FigureWriter(args.output / "figures")
    model_dir = args.output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    results = {
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
    }
    test_scores = {}
    figures.class_distribution(data[TARGET])
    for name in dict.fromkeys(["majority", *args.models]):
        print(f"Training {name}", flush=True)
        classifier = MODEL_FACTORIES[name](args.seed)
        experiment = ChurnExperiment(make_preprocessor(), classifier)
        experiment.fit(splits["train"])
        validation_labels = splits["validation"][TARGET]
        validation_scores = experiment.score(splits["validation"])
        test_scores[name] = experiment.score(splits["test"])
        results["metrics"]["validation"][name] = evaluate(validation_labels, validation_scores)
        results["metrics"]["test"][name] = evaluate(splits["test"][TARGET], test_scores[name])
        results["validation_threshold_probe"]["metrics"][name] = evaluate(
            validation_labels, validation_scores, VALIDATION_PROBE_THRESHOLD,
        )
        example_score = float(experiment.score(pd.DataFrame([EXAMPLE]))[0])
        results["example_churn_probability"][name] = example_score
        joblib.dump(experiment.pipeline, model_dir / f"{name}.joblib", compress=3)
        if hasattr(classifier, "training_loss_"):
            results["ann_training_loss"] = classifier.training_loss_
            figures.training_loss(classifier.training_loss_)

    test_metrics = results["metrics"]["test"]
    figures.precision_recall(splits["test"][TARGET], test_scores, test_metrics)
    figures.confusion_matrices(test_metrics)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, metrics in test_metrics.items():
        print(f"{name}: test F1={metrics['f1']:.3f}, AP={metrics['average_precision']:.3f}")
    print(f"Wrote metrics, figures and models to {args.output.resolve()}")


if __name__ == "__main__":
    main()
