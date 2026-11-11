# Group 55 bank customer churn

## Reproduce the experiment

From this folder, use Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python code/artificial_neural_network.py
```

The default run evaluates validation data only and writes to `results-development/`. Existing run directories are protected from accidental overwrites. The original `results/` directory is retained as historical evidence; its test results have already been inspected. Use `--evaluate-test --output results-reproduction` only for an explicitly planned test evaluation. The default run trains the ANN, a random forest, and a training-prior majority baseline using the same stratified 70/15/15 split and seed 0. The run writes `metrics.json`, `split_manifest.csv`, and raw validation probabilities in its output directory, vector PDFs and 300 dpi PNGs in its `figures/` subdirectory, and fitted pipelines in its `models/` subdirectory. The pipelines accept raw named features and apply the fitted preprocessing during inference.

Run the random forest and baseline alone with a separate output directory:

```sh
.venv/bin/python code/artificial_neural_network.py --models random-forest --output results-rf
```

To add another probabilistic classifier, register a factory in `code/churn_models.py` and pass its key to `--models`. A factory receives the seed and returns a classifier with `fit(X, y)` and `predict_proba(X)` whose two columns correspond to `Exited=0` and `Exited=1`, respectively. For example, import `LogisticRegression` from `sklearn.linear_model` and add `"logistic": lambda seed: LogisticRegression(max_iter=1000, random_state=seed)` to `MODEL_FACTORIES`. The same preprocessing, evaluation, plotting, and persistence will apply.

`ChurnExperiment` accepts the preprocessing pipeline and classifier as dependencies. `ANNClassifier` adapts Keras to the same interface as the scikit-learn estimators. The dataset schema and split live in `churn_data.py`, metrics in `churn_evaluation.py`, and figure generation in `churn_figures.py`.

For single-customer inference after training, run from `code/`:

```python
import joblib
import pandas as pd
from artificial_neural_network import EXAMPLE

pipeline = joblib.load("../results/models/ann.joblib")
probability = pipeline.predict_proba(pd.DataFrame([EXAMPLE]))[0, 1]
print(probability, int(probability > 0.5))
```

## Report

The draft follows the course ACL template. Review the existing contribution statements in `report/main.tex` with both team members, then rebuild `report/main.pdf` with `latexmk -pdf main.tex` or `tectonic main.tex` from `report/`. For Overleaf, upload the contents of `report/` at the project root, plus the `results/figures/` directory and its PDFs. The A2L archive should contain the updated report PDF, dataset, and code. The included CSV is below the 50 MB threshold.


## Extended comparison and checks

```sh
python code/artificial_neural_network.py --models logistic random-forest xgboost --output results-extension
python code/build_comparison_table.py
python -m unittest discover -s code -p 'test_*.py'
```

The recorded extension run and generated table are committed. For a rerun, choose a fresh output directory; never overwrite an earlier experiment. The table generator deliberately reads the recorded `results-extension/metrics.json`.

The progress report preserves the original Keras 12-6-6-1 ANN and seed-0 results. It does not mix these with the earlier assistant-generated scikit-learn ANN/seed-55 experiment. Model files are executable pickle-based artifacts: load only your own trusted models. Install the matching dependencies before loading a pipeline.

The assignment requests the progress report PDF, data and code in one archive. `PROJECT_REPORT.md` describes the whole implementation and its validation limits; it is supporting documentation, not a claim that the final project or independent final evaluation is complete. Use `report/main.pdf` for the course milestone. Do not treat the existing test partition as untouched after its scores have been inspected.


On Windows, create the environment with `py -3.12 -m venv .venv`, install with `.venv\Scripts\python -m pip install -r code/requirements.txt`, and run with `.venv\Scripts\python code/artificial_neural_network.py`. The recorded extension and original run can differ slightly across platforms or dependency builds; keep each run's provenance with its scores.
`results-ann-check/` contains a fresh 100-epoch ANN validation run. It is kept separate from the original historical results.

Row-level split manifests and prediction CSVs are included in the local submission package but are intentionally excluded from this GitHub update. Run the commands above to regenerate them locally. GitHub contains summary metrics, model parameters and provenance. The original repository dataset and historical files are preserved.
