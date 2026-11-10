# wineQual

Predicting red wine quality from physicochemical measurements: regularized regression (ridge, lasso) and a small MLP compared under nested cross-validation.

## Contents

- `baseline.ipynb` — full pipeline: data loading, deduplication, train/test split, nested cross-validation for the training-mean reference, ridge, lasso, and a small MLP, plus the predicted-vs-actual figure. Sections 7–9 add exploratory extensions (second-degree polynomial expansion, extended MLP grid, fold-level paired comparison). Executed outputs are included; runs end to end.
- `data/winequality-red.csv` — UCI Wine Quality (red) dataset: 1,599 rows, 11 physicochemical inputs, one integer quality score.
- `data/winequality.names` — dataset documentation.
- `figures/baseline.png` — out-of-fold predicted vs actual quality for the best two models (primary CV seed).

## Results (development set, repeated 5-fold CV over 5 seeds)

Mean absolute error (MAE), mean across all 25 folds (SD across folds):

| Model | MAE (25-fold mean) | SD |
|---|---|---|
| Training mean | 0.6913 | 0.0291 |
| Ridge | 0.5171 | 0.0221 |
| Lasso | 0.5169 | 0.0219 |
| MLP | 0.5208 | 0.0190 |

The three tuned models are within fold noise of each other. The held-out test set (272 wines) is untouched and reserved for a single final evaluation.

## Exploratory extensions (same folds; test set untouched)

| Extension | MAE (25-fold mean) | SD | Paired improvement vs baseline |
|---|---|---|---|
| Poly-Ridge | 0.5138 | 0.0213 | +0.0033 (SE 0.0014) |
| Poly-Lasso | 0.5125 | 0.0201 | +0.0044 (SE 0.0017) |
| MLP, extended grid | 0.5121 | 0.0215 | +0.0087 (SE 0.0033) |

Paired improvements are fold-level mean differences on the same 25 outer folds (baseline minus extension). They are small relative to the fold spread; a formal paired bootstrap is deferred to the draft.

## Running

```bash
python3 -m venv .venv
.venv/bin/pip install pandas scikit-learn matplotlib nbconvert ipykernel
.venv/bin/python -m nbconvert --to notebook --execute baseline.ipynb --output baseline.ipynb
```

## Data

[UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), licensed CC BY 4.0. Please cite: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4):547-553, 2009.
