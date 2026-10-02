# Telco Customer Churn: signals and a risk model

Analysis behind the DataStories.fyi story
[I Never Saw It Coming](https://datastories.fyi/i-never-saw-it-coming.html).

## Data
IBM Telco Customer Churn sample dataset (7,043 customers, 1,869 churned).
Not included in this repo. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and place it in a `Data/` folder.

## Files
- `check_story.py`- recomputes every number mentioned in the story.
- `model.py`: logistic regression and gradient boosting churn models, risk-decile table, 5-fold cross-validation.

## Method
80/20 stratified train/test split (random_state=42). Numeric features scaled,
categorical features one-hot encoded. `TotalCharges` dropped because it is
roughly tenure x monthly charges.

## Results
| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic regression | 0.839 | 0.630 |
| Gradient boosting | 0.833 | 0.641 |

5-fold cross-validated ROC-AUC (logistic): 0.843 (range 0.833 to 0.857).
Top 20% of customers by predicted risk contain 49.5% of churners (test set, n=1,409).

## Limits
The model ranks risk. It does not establish causes. The data is a sample dataset published by IBM, not a live customer base.

## Run it
    pip install -r requirements.txt
    python check_story.py
    python model.py
