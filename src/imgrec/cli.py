"""``imgrec`` command line: run the full experiment and write results."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from . import plots
from .data import OPENML_NAMES, load
from .experiment import run
from .models import model_specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imgrec", description="Tune, compare and stress-test image classifiers.")
    parser.add_argument("--dataset", choices=["digits", *OPENML_NAMES], default="digits")
    parser.add_argument("--models", nargs="+", choices=sorted(model_specs()), default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="stratified subsample (useful for MNIST)")
    parser.add_argument("--quick", action="store_true", help="small grids and 3-fold CV (for CI / smoke tests)")
    parser.add_argument("--out", default="results")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    ds = load(args.dataset, max_samples=args.max_samples, seed=args.seed)
    print(
        f"{ds.name}: {len(ds.X_train):,} train / {len(ds.X_test):,} test images, {ds.image_shape[0]}x{ds.image_shape[1]}, {ds.n_classes} classes"
    )
    result = run(ds, models=args.models, quick=args.quick, seed=args.seed)

    out = result.save(Path(args.out) / ds.name)
    joblib.dump(result.fitted[result.best_model], out / "best_model.joblib")
    if not args.no_figures:
        plots.all_figures(result, ds, out / "figures")
    print()
    print(result.to_markdown())
    print(f"Results written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
