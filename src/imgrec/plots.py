"""Figures written to ``results/figures``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .data import ImageDataset  # noqa: E402
from .experiment import ExperimentResult  # noqa: E402


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def sample_grid(ds: ImageDataset, path: Path, per_class: int = 6) -> Path:
    fig, axes = plt.subplots(ds.n_classes, per_class, figsize=(per_class * 0.9, ds.n_classes * 0.9))
    for c in range(ds.n_classes):
        idx = np.flatnonzero(ds.y_train == c)[:per_class]
        for j, ax in enumerate(axes[c]):
            ax.axis("off")
            if j < len(idx):
                ax.imshow(ds.X_train[idx[j]].reshape(ds.image_shape), cmap="gray_r")
        axes[c][0].set_title(ds.class_names[c], fontsize=7, loc="left")
    return _save(fig, path)


def confusion(result: ExperimentResult, ds: ImageDataset, path: Path) -> Path:
    cm = result.confusion
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(ds.n_classes), ds.class_names, rotation=45, ha="right")
    ax.set_yticks(range(ds.n_classes), ds.class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - {result.best_model}")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > thresh else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def misclassified(result: ExperimentResult, ds: ImageDataset, path: Path, limit: int = 16) -> Path:
    model = result.fitted[result.best_model]
    idx = result.misclassified[:limit]
    cols = 8
    rows = max(1, int(np.ceil(len(idx) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.45), squeeze=False)
    preds = model.predict(ds.X_test[idx]) if len(idx) else []
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i, p in zip(axes.ravel()[: len(idx)], idx, preds, strict=True):
        ax.imshow(ds.X_test[i].reshape(ds.image_shape), cmap="gray_r")
        ax.set_title(f"true {ds.class_names[ds.y_test[i]]}\npred {ds.class_names[p]}", fontsize=7)
    fig.suptitle(f"Misclassified test images ({len(result.misclassified)} of {len(ds.y_test)})", fontsize=9)
    return _save(fig, path)


def robustness(result: ExperimentResult, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in result.robustness.columns:
        ax.plot(result.robustness.index, result.robustness[name], marker="o", label=name)
    ax.set_xlabel("Gaussian noise sigma (pixels scaled to [0, 1])")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Robustness to pixel noise")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def pca_variance(ds: ImageDataset, path: Path) -> Path:
    pca = PCA().fit(StandardScaler().fit_transform(ds.X_train))
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(cumulative) + 1), cumulative)
    ax.axhline(0.95, ls="--", color="grey", lw=1)
    ax.axvline(n95, ls="--", color="grey", lw=1)
    ax.annotate(f"{n95} components explain 95%", (n95, 0.95), xytext=(n95 + 3, 0.75), fontsize=9)
    ax.set_xlabel("Principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA on standardised pixels")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def all_figures(result: ExperimentResult, ds: ImageDataset, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    return [
        sample_grid(ds, out / "samples.png"),
        confusion(result, ds, out / "confusion_matrix.png"),
        misclassified(result, ds, out / "misclassified.png"),
        robustness(result, out / "robustness.png"),
        pca_variance(ds, out / "pca_variance.png"),
    ]
