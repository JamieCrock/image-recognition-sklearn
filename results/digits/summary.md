## Model comparison (digits)

| model | cv_accuracy | test_accuracy | macro_f1 | fit_seconds |
|---|---|---|---|---|
| svm_rbf | 99.26% | 99.11% | 0.9911 | 0.04 |
| knn | 98.81% | 98.44% | 0.9842 | 0.00 |
| random_forest | 97.85% | 96.67% | 0.9664 | 0.45 |
| logistic_regression | 96.73% | 96.44% | 0.9640 | 0.02 |
| mlp | 96.81% | 95.33% | 0.9529 | 0.46 |

Best model: **svm_rbf** with `{'clf__C': 1, 'clf__gamma': 'scale', 'pca__n_components': 40, 'scale': 'passthrough'}`

## Augmentation (svm_rbf)

| Training data | Test accuracy |
|---|---|
| Original (1,347 images) | 99.11% |
| + shifted copies (6,735 images) | 99.56% |

## Robustness to Gaussian noise (test accuracy)

| model | sigma=0.0 | sigma=0.05 | sigma=0.1 | sigma=0.2 | sigma=0.3 | sigma=0.4 |
|---|---|---|---|---|---|---|
| logistic_regression | 96.4% | 93.1% | 83.3% | 55.6% | 34.9% | 28.7% |
| knn | 98.4% | 98.4% | 98.0% | 96.9% | 94.7% | 90.2% |
| svm_rbf | 99.1% | 98.4% | 98.2% | 97.1% | 94.4% | 84.2% |
| random_forest | 96.7% | 96.2% | 95.3% | 92.0% | 84.0% | 75.3% |
| mlp | 95.3% | 46.9% | 30.2% | 19.8% | 16.0% | 14.0% |

## Scaling ablation (test accuracy, noise sigma=0.1)

| model | clean, standardised | noisy, standardised | clean, raw pixels | noisy, raw pixels |
|---|---|---|---|---|
| logistic_regression | 96.4% | 83.3% | 96.7% | 95.8% |
| knn | 96.9% | 33.3% | 98.4% | 98.0% |
| svm_rbf | 98.2% | 13.6% | 99.1% | 98.2% |
| mlp | 95.3% | 30.2% | 94.9% | 95.3% |
