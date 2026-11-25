import json

import numpy as np
import pytest

from imgrec import add_noise, augment, load, model_specs, run
from imgrec.cli import main


@pytest.fixture(scope="module")
def digits():
    return load("digits", seed=0)


@pytest.fixture(scope="module")
def quick_result(digits):
    return run(digits, models=["logistic_regression", "knn", "svm_rbf"], quick=True, seed=0, verbose=False)


def test_digits_loads_scaled_and_stratified(digits):
    assert digits.X_train.shape[1] == 64
    assert digits.image_shape == (8, 8)
    assert 0.0 <= digits.X_train.min() and digits.X_train.max() <= 1.0
    assert len(digits.X_train) + len(digits.X_test) == 1797
    train_share = np.bincount(digits.y_train) / len(digits.y_train)
    test_share = np.bincount(digits.y_test) / len(digits.y_test)
    assert np.allclose(train_share, test_share, atol=0.01)


def test_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load("cifar")


def test_augment_shapes_and_labels(digits):
    X, y = augment(digits.X_train[:10], digits.y_train[:10], digits.image_shape)
    assert X.shape == (50, 64)
    assert np.array_equal(y, np.tile(digits.y_train[:10], 5))
    assert np.array_equal(X[:10], digits.X_train[:10])  # originals preserved


def test_shift_moves_pixels():
    img = np.zeros((1, 9))
    img[0, 4] = 1.0  # centre of a 3x3 image
    X, _ = augment(img, np.array([0]), (3, 3))
    shifted = X[1:].reshape(4, 3, 3)
    assert all(s.sum() == 1.0 for s in shifted)
    assert not any(np.array_equal(s, img.reshape(3, 3)) for s in shifted)


def test_augment_with_rotation_stays_in_range(digits):
    X, _ = augment(digits.X_train[:5], digits.y_train[:5], digits.image_shape, max_rotation=10)
    assert X.min() >= 0 and X.max() <= 1


def test_noise_is_clipped_and_reproducible(digits):
    a = add_noise(digits.X_test, 0.3, seed=1)
    b = add_noise(digits.X_test, 0.3, seed=1)
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() <= 1
    assert np.array_equal(add_noise(digits.X_test, 0), digits.X_test)


def test_every_model_spec_has_grids():
    for spec in model_specs().values():
        assert spec.grid and spec.quick_grid
        for key in spec.quick_grid:
            assert key in spec.pipeline.get_params()


def test_quick_run_is_accurate(quick_result):
    assert quick_result.comparison["test_accuracy"].max() > 0.95
    assert quick_result.best_model in {"logistic_regression", "knn", "svm_rbf"}
    assert quick_result.confusion.shape == (10, 10)
    assert quick_result.confusion.trace() == pytest.approx(
        quick_result.comparison.loc[quick_result.best_model, "test_accuracy"] * quick_result.confusion.sum()
    )


def test_noise_degrades_accuracy(quick_result):
    rob = quick_result.robustness
    assert (rob.iloc[-1] < rob.iloc[0]).all()


def test_augmentation_result_recorded(quick_result):
    aug = quick_result.augmentation
    assert aug["n_train_augmented"] == 5 * aug["n_train_original"]
    assert 0.9 < aug["accuracy_augmented"] <= 1.0


def test_cli_writes_results(tmp_path):
    assert main(["--quick", "--models", "logistic_regression", "knn", "--out", str(tmp_path)]) == 0
    out = tmp_path / "digits"
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["test_accuracy"] > 0.9
    for name in ("model_comparison.csv", "robustness.csv", "summary.md", "best_model.joblib"):
        assert (out / name).exists()
    figures = sorted(p.name for p in (out / "figures").iterdir())
    assert figures == ["confusion_matrix.png", "misclassified.png", "pca_variance.png", "robustness.png", "samples.png"]


def test_scaling_ablation_shows_standardisation_amplifies_noise(quick_result):
    table = quick_result.scaling
    assert set(table.index) == {"logistic_regression", "knn", "svm_rbf"}
    assert list(table.columns) == [
        "clean, standardised",
        "noisy, standardised",
        "clean, raw pixels",
        "noisy, raw pixels",
    ]
    # Near-constant border pixels get blown up by StandardScaler, so noise hurts far more.
    assert (table["noisy, raw pixels"] > table["noisy, standardised"]).all()
