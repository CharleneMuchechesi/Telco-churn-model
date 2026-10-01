from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CSV = Path(__file__).parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn", "customerID", "TotalCharges"])

num = ["tenure", "MonthlyCharges"]
cat = [column for column in X.columns if column not in num]

prep = ColumnTransformer(
	[
		("num", StandardScaler(), num),
		("cat", OneHotEncoder(handle_unknown="ignore"), cat),
	],
	# HistGradientBoostingClassifier requires dense input.
	sparse_threshold=0,
)

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
	"logistic": LogisticRegression(max_iter=1000),
	"gradient boosting": HistGradientBoostingClassifier(random_state=42),
}

for name, clf in models.items():
	pipe = Pipeline([("prep", prep), ("clf", clf)]).fit(X_train, y_train)
	probabilities = pipe.predict_proba(X_test)[:, 1]
	print(
		name,
		"ROC-AUC:",
		round(roc_auc_score(y_test, probabilities), 3),
		"| PR-AUC:",
		round(average_precision_score(y_test, probabilities), 3),
	)

pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
p = pipe.predict_proba(X_test)[:, 1]

res = pd.DataFrame({"p": p, "y": y_test.values})
res["decile"] = pd.qcut(res["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
tab = res.groupby("decile", observed=True).agg(customers=("y", "size"), churners=("y", "sum"))
tab["share_of_churners"] = tab["churners"] / res["y"].sum()
tab["churn_rate"] = tab["churners"] / tab["customers"]
print(tab.round(3))
print("Top 20% of customers by risk capture", tab["share_of_churners"].head(2).sum().round(3), "of churners")

names = pipe.named_steps["prep"].get_feature_names_out()
coef = pd.Series(pipe.named_steps["clf"].coef_[0], index=names).sort_values()
print(coef.head(8)); print(coef.tail(8))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))]),
                         X, y, cv=5, scoring="roc_auc")
print(scores.round(3), "mean:", scores.mean().round(3))