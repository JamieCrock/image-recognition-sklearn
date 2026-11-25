"""End-to-end experiment: tune, compare, stress-test and report.

1. **Model selection** - every candidate is tuned with ``GridSearchCV`` over
   stratified 5-fold cross-validation on the training set only.
2. **Held-out evaluation** - the tuned models are scored once on the untouched
   test set (accuracy, macro precision/recall/F1, fit time).
3. **Augmentation study** - the best model is retrained on shift-augmented
   data and re-scored.
4. **Robustness** - test images are corrupted with increasing Gaussian noise
   to see how gracefully each model degrades.
5. **Scaling ablation** - each tuned model is refitted with and without pixel
   standardisation to isolate its effect on clean and noisy accuracy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .data import ImageDataset, add_noise, augment
from .models import model_specs

NOISE_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
ABLATION_SIGMA = 0.1


def markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame (index included) as a GitHub-flavoured Markdown table."""
    header = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


@dataclass
class ExperimentResult:
    dataset: str
    comparison: pd.DataFrame
    best_model: str
    best_params: dict
    confusion: np.ndarray
    report: dict
    augmentation: dict
    robustness: pd.DataFrame
    scaling: pd.DataFrame
    fitted: dict = field(default_factory=dict, repr=False)
    misclassified: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int), repr=False)

    def to_markdown(self) -> str:
        cols = ["cv_accuracy", "test_accuracy", "macro_f1", "fit_seconds"]
        table = self.comparison[cols].copy()
        table["cv_accuracy"] = table["cv_accuracy"].map("{:.2%}".format)
        table["test_accuracy"] = table["test_accuracy"].map("{:.2%}".format)
        table["macro_f1"] = table["macro_f1"].map("{:.4f}".format)
        table["fit_seconds"] = table["fit_seconds"].map("{:.2f}".format)
        rob = self.robustness.copy()
        rob.index = pd.Index([f"sigma={s}" for s in rob.index])
        rob.columns.name = "model"
        rob = rob.map("{:.1%}".format)
        aug = self.augmentation
        return (
            f"## Model comparison ({self.dataset})\n\n{markdown_table(table)}\n\n"
            f"Best model: **{self.best_model}** with `{self.best_params}`\n\n"
            f"## Augmentation ({self.best_model})\n\n"
            f"| Training data | Test accuracy |\n|---|---|\n"
            f"| Original ({aug['n_train_original']:,} images) | {aug['accuracy_original']:.2%} |\n"
            f"| + shifted copies ({aug['n_train_augmented']:,} images) | {aug['accuracy_augmented']:.2%} |\n\n"
            f"## Robustness to Gaussian noise (test accuracy)\n\n{markdown_table(rob.T)}\n\n"
            f"## Scaling ablation (test accuracy, noise sigma={ABLATION_SIGMA})\n\n"
            f"{markdown_table(self.scaling.map('{:.1%}'.format))}\n"
        )

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.comparison.to_csv(out / "model_comparison.csv")
        self.robustness.to_csv(out / "robustness.csv")
        self.scaling.to_csv(out / "scaling_ablation.csv")
        (out / "summary.md").write_text(self.to_markdown(), encoding="utf-8")
        payload = {
            "dataset": self.dataset,
            "best_model": self.best_model,
            "best_params": {
                k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in self.best_params.items()
            },
            "test_accuracy": float(self.comparison.loc[self.best_model, "test_accuracy"]),
            "augmentation": self.augmentation,
            "per_class": {k: v for k, v in self.report.items() if k not in {"accuracy"}},
        }
        (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out


def tune(name: str, ds: ImageDataset, quick: bool, seed: int, n_jobs: int = -1) -> GridSearchCV:
    spec = model_specs(seed)[name]
    cv = StratifiedKFold(n_splits=3 if quick else 5, shuffle=True, random_state=seed)
    search = GridSearchCV(
        spec.pipeline, spec.quick_grid if quick else spec.grid, cv=cv, n_jobs=n_jobs, scoring="accuracy"
    )
    search.fit(ds.X_train, ds.y_train)
    return search


def run(
    ds: ImageDataset, models: list[str] | None = None, quick: bool = False, seed: int = 42, verbose: bool = True
) -> ExperimentResult:
    names = models or list(model_specs(seed))
    rows, fitted = [], {}
    for name in names:
        start = time.perf_counter()
        search = tune(name, ds, quick, seed)
        elapsed = time.perf_counter() - start
        best = search.best_estimator_
        # Time a single refit with the chosen hyperparameters for a fair comparison.
        t0 = time.perf_counter()
        best = clone(best).fit(ds.X_train, ds.y_train)
        fit_seconds = time.perf_counter() - t0
        y_pred = best.predict(ds.X_test)
        p, r, f1, _ = precision_recall_fscore_support(ds.y_test, y_pred, average="macro", zero_division=0)
        rows.append(
            {
                "model": name,
                "cv_accuracy": search.best_score_,
                "test_accuracy": accuracy_score(ds.y_test, y_pred),
                "macro_precision": p,
                "macro_recall": r,
                "macro_f1": f1,
                "fit_seconds": fit_seconds,
                "search_seconds": elapsed,
                "best_params": json.dumps(search.best_params_, default=str),
            }
        )
        fitted[name] = best
        if verbose:
            print(f"  {name:<20} cv={search.best_score_:.4f}  test={rows[-1]['test_accuracy']:.4f}  ({elapsed:.1f}s)")

    comparison = pd.DataFrame(rows).set_index("model").sort_values(["test_accuracy", "cv_accuracy"], ascending=False)
    best_name = comparison.index[0]
    best_model = fitted[best_name]
    y_pred = best_model.predict(ds.X_test)

    X_aug, y_aug = augment(ds.X_train, ds.y_train, ds.image_shape, seed=seed)
    aug_model = clone(best_model).fit(X_aug, y_aug)
    augmentation = {
        "model": best_name,
        "n_train_original": int(len(ds.X_train)),
        "n_train_augmented": int(len(X_aug)),
        "accuracy_original": float(accuracy_score(ds.y_test, y_pred)),
        "accuracy_augmented": float(accuracy_score(ds.y_test, aug_model.predict(ds.X_test))),
    }

    robustness = pd.DataFrame(
        {
            name: [accuracy_score(ds.y_test, model.predict(add_noise(ds.X_test, s, seed))) for s in NOISE_LEVELS]
            for name, model in fitted.items()
        },
        index=pd.Index(NOISE_LEVELS, name="sigma"),
    )

    scaling = scaling_ablation(fitted, ds, seed)

    return ExperimentResult(
        dataset=ds.name,
        comparison=comparison,
        best_model=best_name,
        best_params=json.loads(comparison.loc[best_name, "best_params"]),
        confusion=confusion_matrix(ds.y_test, y_pred),
        report=classification_report(
            ds.y_test, y_pred, target_names=list(ds.class_names), output_dict=True, zero_division=0
        ),
        augmentation=augmentation,
        robustness=robustness,
        scaling=scaling,
        fitted=fitted,
        misclassified=np.flatnonzero(y_pred != ds.y_test),
    )


def scaling_ablation(fitted: dict, ds: ImageDataset, seed: int = 42) -> pd.DataFrame:
    """Refit each tuned model with and without standardisation; score clean and noisy."""
    from sklearn.preprocessing import StandardScaler

    noisy = add_noise(ds.X_test, ABLATION_SIGMA, seed)
    rows = []
    for name, model in fitted.items():
        if "scale" not in model.named_steps:
            continue
        row = {"model": name}
        for label, scaler in (("standardised", StandardScaler()), ("raw pixels", "passthrough")):
            variant = clone(model).set_params(scale=scaler).fit(ds.X_train, ds.y_train)
            row[f"clean, {label}"] = accuracy_score(ds.y_test, variant.predict(ds.X_test))
            row[f"noisy, {label}"] = accuracy_score(ds.y_test, variant.predict(noisy))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
