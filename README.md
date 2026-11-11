# Group 55 bank customer churn

## Reproduce the experiment

From this folder, use Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python code/artificial_neural_network.py
```

The default run trains the ANN, a random forest, and a training-prior majority baseline using the same stratified 70/15/15 split and seed 0. The run writes `results/metrics.json`, vector PDFs and 300 dpi PNGs in `results/figures/`, and fitted pipelines in `results/models/`. The pipelines accept raw named features and apply the fitted preprocessing during inference.

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

The draft follows the course ACL template. Update the marked contribution entry in `report/main.tex` with verified details, then rebuild `report/main.pdf` with `latexmk -pdf main.tex` or `tectonic main.tex` from `report/`. For Overleaf, upload the contents of `report/` at the project root, plus the `results/figures/` directory and its PDFs. The A2L archive should contain the updated report PDF, dataset, and code. The included CSV is below the 50 MB threshold.
