.PHONY: install test lint run quick mnist fashion clean

PY ?= python3

install:      ## install with dev tools
	$(PY) -m pip install -e ".[dev]"

test:         ## pytest with coverage
	$(PY) -m pytest --cov=imgrec --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

run:          ## full experiment on the bundled digits dataset (~1 min)
	imgrec --dataset digits

quick:        ## small grids, 3-fold CV
	imgrec --dataset digits --quick

mnist:        ## MNIST via OpenML (downloads ~50 MB on first run)
	imgrec --dataset mnist --max-samples 20000 --quick

fashion:      ## Fashion-MNIST via OpenML
	imgrec --dataset fashion-mnist --max-samples 20000 --quick

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
