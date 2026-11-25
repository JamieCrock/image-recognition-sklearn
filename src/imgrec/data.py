"""Image datasets and augmentation.

``digits`` (1,797 greyscale 8x8 handwritten digits) ships with scikit-learn, so
the whole project runs offline. ``mnist`` and ``fashion-mnist`` (70,000 28x28
images each) are fetched from OpenML on first use and cached by scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import rotate, shift
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split

OPENML_NAMES = {"mnist": "mnist_784", "fashion-mnist": "Fashion-MNIST"}
FASHION_LABELS = (
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)  # fmt: skip


@dataclass(frozen=True)
class ImageDataset:
    name: str
    X_train: np.ndarray  # (n, h*w) floats in [0, 1]
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    image_shape: tuple[int, int]
    class_names: tuple[str, ...]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def load(name: str = "digits", test_size: float = 0.25, max_samples: int | None = None, seed: int = 42) -> ImageDataset:
    """Load a dataset, scale pixels to [0, 1] and make a stratified split."""
    if name == "digits":
        bunch = load_digits()
        X, y = bunch.data / 16.0, bunch.target
        shape = (8, 8)
        class_names = tuple(str(c) for c in range(10))
    elif name in OPENML_NAMES:
        X, y = fetch_openml(OPENML_NAMES[name], version=1, return_X_y=True, as_frame=False, parser="auto")
        X, y = X / 255.0, y.astype(int)
        shape = (28, 28)
        class_names = FASHION_LABELS if name == "fashion-mnist" else tuple(str(c) for c in range(10))
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: digits, {', '.join(OPENML_NAMES)}")

    if max_samples is not None and len(X) > max_samples:
        X, _, y, _ = train_test_split(X, y, train_size=max_samples, stratify=y, random_state=seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    return ImageDataset(name, X_train, X_test, y_train, y_test, shape, class_names)


def augment(
    X: np.ndarray,
    y: np.ndarray,
    image_shape: tuple[int, int],
    shifts: int = 1,
    max_rotation: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Add copies of every image shifted one pixel up/down/left/right.

    ``shifts`` is the pixel offset; with ``max_rotation > 0`` each shifted copy
    is also rotated by a random angle in ``[-max_rotation, max_rotation]``.
    Returns the original data followed by the augmented copies (5x the rows).
    """
    rng = np.random.default_rng(seed)
    images = X.reshape(-1, *image_shape)
    copies = [images]
    for dy, dx in ((shifts, 0), (-shifts, 0), (0, shifts), (0, -shifts)):
        moved = shift(images, (0, dy, dx), order=0, mode="constant", cval=0.0)
        if max_rotation:
            angles = rng.uniform(-max_rotation, max_rotation, size=len(moved))
            moved = np.stack(
                [rotate(img, a, reshape=False, order=1, mode="constant") for img, a in zip(moved, angles, strict=True)]
            )
        copies.append(moved)
    X_aug = np.concatenate(copies).reshape(-1, X.shape[1])
    y_aug = np.tile(y, len(copies))
    return np.clip(X_aug, 0.0, 1.0), y_aug


def add_noise(X: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Gaussian pixel noise, clipped back to [0, 1]."""
    if sigma <= 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    return np.clip(X + rng.normal(0.0, sigma, size=X.shape), 0.0, 1.0)
