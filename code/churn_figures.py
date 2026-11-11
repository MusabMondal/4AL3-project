"""Export vector PDFs for LaTeX and 300 dpi PNGs for other editors."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

LABELS = {"majority": "Majority baseline", "ann": "ANN", "random-forest": "Random forest"}


class FigureWriter:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name: str) -> None:
        for extension in ("pdf", "png"):
            fig.savefig(self.directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def class_distribution(self, labels: pd.Series) -> None:
        counts = labels.value_counts().reindex([0, 1])
        fig, ax = plt.subplots(figsize=(4.1, 2.65), layout="constrained")
        bars = ax.bar(["Stayed (0)", "Exited (1)"], counts, color=["#48778D", "#C56432"], width=0.6)
        ax.bar_label(bars, labels=[f"{n:,} ({n / len(labels):.1%})" for n in counts], padding=4)
        ax.set(ylabel="Customers", ylim=(0, max(counts) * 1.22))
        ax.spines[["top", "right"]].set_visible(False)
        self._save(fig, "class_distribution")

    def precision_recall(self, labels: pd.Series, scores: dict, metrics: dict) -> None:
        fig, ax = plt.subplots(figsize=(4.35, 3.15), layout="constrained")
        ax.axhline(labels.mean(), color="#6E7177", linestyle="--",
                   label=f"No skill (AP {labels.mean():.3f})")
        for name, values in scores.items():
            if name == "majority":
                continue
            precision, recall, _ = precision_recall_curve(labels, values)
            ap = metrics[name]["average_precision"]
            ax.step(recall, precision, where="post", linewidth=1.65,
                    label=f"{LABELS.get(name, name)} (AP {ap:.3f})")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.02), ylim=(0, 1.03))
        ax.grid(alpha=0.22)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
        self._save(fig, "test_precision_recall")

    def confusion_matrices(self, metrics: dict) -> None:
        fig, axes = plt.subplots(1, len(metrics), figsize=(2.4 * len(metrics), 2.35),
                                 squeeze=False, layout="constrained")
        for ax, (name, result) in zip(axes[0], metrics.items()):
            counts = np.array(result["confusion_matrix"])
            fractions = counts / counts.sum(axis=1, keepdims=True)
            ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
            for actual, predicted in np.ndindex(counts.shape):
                fraction = fractions[actual, predicted]
                ax.text(predicted, actual, f"{counts[actual, predicted]:,}\n{fraction:.0%}",
                        ha="center", va="center", color="white" if fraction > 0.55 else "#132734")
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Stayed", "Exited"],
                   yticklabels=["Stayed", "Exited"], xlabel="Predicted", title=LABELS.get(name, name))
        axes[0, 0].set_ylabel("Actual")
        self._save(fig, "test_confusion_matrices")

    def training_loss(self, loss: list[float]) -> None:
        fig, ax = plt.subplots(figsize=(4.35, 2.75), layout="constrained")
        ax.plot(range(1, len(loss) + 1), loss, color="#235E82", linewidth=1.75)
        ax.set(xlabel="Epoch", ylabel="Training binary cross-entropy")
        ax.grid(alpha=0.22)
        self._save(fig, "ann_training_loss")
