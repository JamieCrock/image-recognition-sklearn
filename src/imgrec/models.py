"""Candidate classifiers and their hyperparameter grids.

Every model is a ``Pipeline`` so scaling and dimensionality reduction are
fitted inside each cross-validation fold - no information leaks from the
validation folds into preprocessing.

Whether to standardise pixels is itself a hyperparameter (the ``scale`` step).
Pixels already share a [0, 1] range, and standardising near-constant border
pixels turns out to amplify noise badly - see the scaling ablation in
``experiment.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _scale_options() -> list:
    return [StandardScaler(), "passthrough"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    pipeline: Pipeline
    grid: dict[str, list] = field(default_factory=dict)
    quick_grid: dict[str, list] = field(default_factory=dict)


def model_specs(seed: int = 42) -> dict[str, ModelSpec]:
    """Fresh, unfitted pipelines with full and quick (CI-sized) search grids."""
    specs = [
        ModelSpec(
            "logistic_regression",
            Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=seed))]),
            grid={"scale": _scale_options(), "clf__C": [0.01, 0.1, 1.0, 10.0]},
            quick_grid={"clf__C": [0.1, 1.0]},
        ),
        ModelSpec(
            "knn",
            Pipeline([("scale", StandardScaler()), ("pca", PCA(random_state=seed)), ("clf", KNeighborsClassifier())]),
            grid={
                "scale": _scale_options(),
                "pca__n_components": [20, 30, 40],
                "clf__n_neighbors": [1, 3, 5, 7],
                "clf__weights": ["uniform", "distance"],
            },
            quick_grid={"pca__n_components": [30], "clf__n_neighbors": [3, 5]},
        ),
        ModelSpec(
            "svm_rbf",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(random_state=seed)),
                    ("clf", SVC(kernel="rbf", random_state=seed)),
                ]
            ),
            grid={
                "scale": _scale_options(),
                "pca__n_components": [30, 40, None],
                "clf__C": [1, 5, 10, 50],
                "clf__gamma": ["scale", 0.01, 0.02],
            },
            quick_grid={"pca__n_components": [40], "clf__C": [5, 10]},
        ),
        ModelSpec(
            "random_forest",
            Pipeline([("clf", RandomForestClassifier(n_jobs=-1, random_state=seed))]),
            grid={"clf__n_estimators": [200, 500], "clf__max_features": ["sqrt", 0.2], "clf__min_samples_leaf": [1, 2]},
            quick_grid={"clf__n_estimators": [150]},
        ),
        ModelSpec(
            "mlp",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", MLPClassifier(max_iter=600, early_stopping=True, n_iter_no_change=20, random_state=seed)),
                ]
            ),
            grid={
                "scale": _scale_options(),
                "clf__hidden_layer_sizes": [(128,), (256,), (256, 128)],
                "clf__alpha": [1e-4, 1e-3, 1e-2],
            },
            quick_grid={"clf__hidden_layer_sizes": [(128,)], "clf__alpha": [1e-3]},
        ),
    ]
    return {s.name: s for s in specs}
