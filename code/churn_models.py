"""The Keras adapter exposes the same probability interface as scikit-learn."""

from typing import Protocol, Self

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted


class ProbabilisticClassifier(Protocol):
    """Train on transformed data and return two columns ordered as 0, 1."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class ANNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, epochs: int = 100, batch_size: int = 32, random_state: int = 0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.random_state)
        tf.config.experimental.enable_op_determinism()
        network = tf.keras.Sequential([
            tf.keras.Input(shape=(X.shape[1],)),
            tf.keras.layers.Dense(6, activation="relu"),
            tf.keras.layers.Dense(6, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        network.compile(optimizer="adam", loss="binary_crossentropy")
        history = network.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        self.model_ = network
        self.classes_ = np.array([0, 1])
        self.training_loss_ = history.history["loss"]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        positive = self.model_.predict(X, verbose=0).ravel()
        return np.column_stack((1 - positive, positive))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# Only this registry needs another entry to offer a new classifier in the CLI.
MODEL_FACTORIES = {
    "majority": lambda seed: DummyClassifier(strategy="prior"),
    "ann": lambda seed: ANNClassifier(random_state=seed),
    "random-forest": lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed),
}
