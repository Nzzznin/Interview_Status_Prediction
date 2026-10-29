# Interview Status Prediction

Predicting the outcome of a job interview — **Consider**, **May Consider** or **Not Consider** — from interview behaviour and engagement metrics, using classic machine-learning classifiers.

The whole pipeline (cleaning → EDA → feature analysis → model training → tuning → evaluation) lives in a single notebook: [`main.ipynb`](main.ipynb).

## Results

Six classifiers were trained on an 80/20 split, tuned with `GridSearchCV`, and then validated with 5-fold stratified cross-validation.

| Model | Test accuracy (tuned) | F1 (weighted) | ROC-AUC (OvR) | 5-fold CV accuracy |
|---|---|---|---|---|
| **Gradient Boosting** | **0.960** | **0.961** | **0.995** | **0.955 ± 0.006** |
| Random Forest | 0.919 | 0.918 | 0.987 | 0.907 ± 0.009 |
| AdaBoost | 0.886 | 0.889 | 0.920 | 0.888 ± 0.008 |
| Decision Tree | 0.877 | 0.877 | 0.904 | 0.875 ± 0.010 |
| KNN | 0.870 | 0.870 | 0.967 | 0.861 ± 0.005 |
| Naïve Bayes | 0.832 | 0.833 | 0.951 | 0.831 ± 0.010 |

**Gradient Boosting** is the best model on every metric.

## Dataset

Located in [`dataset/`](dataset):

| File | Rows | Description |
|---|---|---|
| `train.csv` | 5,800 | Features + `Status` label |
| `test.csv` | 1,200 | Features only |
| `result.csv` | 1,200 | `Interview Id` → `Status` for the test set |

Class distribution in the training set: Consider (2,311), May Consider (1,890), Not Consider (1,595).

Each row describes one interview with 26 columns, including:

- **IDs** — `Interview Id`, `Candidate Id`, `Interviewer Id` (dropped before training)
- **Profile** — the role applied for (e.g. Developer, Sales, Product Manager)
- **Talk-time and engagement metrics** — abbreviated numeric features such as `S.L.R.C`, `S.L.R.I`, `A.T.T`, `L.J.T.C`, `N.I.C`, `S.P.C`, `Q.A`, …
- **Sentiment** — `P.E.I` / `P.E.C` (Positive / Neutral / Negative)
- **Interview flow** — `COMPLIANCE Ratio`, `Interview duration`, `Interviewer Intro`, `Candidate into`, `Opp to ask`

## Pipeline

1. **Data cleaning** — fill missing values (mode for categorical, mean for numeric), remove duplicates, IQR outlier filtering, drop ID and constant columns, normalise text.
2. **Exploratory analysis** — distribution plots for every numeric feature, class-balance plot, before/after cleaning comparison.
3. **Encoding** — `LabelEncoder` fitted on the training set and reused on the test set so encodings stay consistent.
4. **Feature analysis** — correlation with the target (`L.J.T.C` is by far the strongest, r ≈ 0.86) and Random Forest feature importance.
5. **Modelling** — Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Gaussian Naïve Bayes and KNN.
6. **Hyperparameter tuning** — `GridSearchCV` (5-fold) per model.
7. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrices, default-vs-tuned comparison and K-fold cross-validation.

## Getting started

```bash
git clone https://github.com/Nzzznin/Interview_Status_Prediction.git
cd Interview_Status_Prediction

python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
jupyter lab main.ipynb
```

The core libraries are `pandas`, `numpy`, `scikit-learn`, `matplotlib` and `seaborn`.

## Project structure

```
Interview_Status_Prediction/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── result.csv
├── main.ipynb          # Full analysis and modelling notebook
└── requirements.txt
```
