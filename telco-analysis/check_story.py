from pathlib import Path
import pandas as pd

CSV = Path(__file__).parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks in the file
df["churn"] = (df["Churn"] == "Yes").astype(int)

print("Customers:", len(df), "| churned:", df["churn"].sum(), "| rate:", df["churn"].mean().round(3))
print("Avg monthly charge of churners:", df.loc[df.churn == 1, "MonthlyCharges"].mean().round(2))
print("Monthly revenue lost:", df.loc[df.churn == 1, "MonthlyCharges"].sum().round(0))

for col in ["Contract", "PaymentMethod", "InternetService"]:
    print(df.groupby(col)["churn"].agg(["mean", "size"]).round(3), "\n")

# The pull-quote problem: two different numbers
first3 = df[df.tenure <= 3]
print("Churn rate among customers in first 3 months:", first3["churn"].mean().round(3))
print("Share of ALL churners who left in first 3 months:", (df.loc[df.churn == 1, "tenure"] <= 3).mean().round(3))
print("Churn rate at 60+ months:", df.loc[df.tenure >= 60, "churn"].mean().round(3))

# Number of services (adjust this list to match how you defined it originally)
services = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
df["n_services"] = (df[services] == "Yes").sum(axis=1)
print(df.groupby("n_services")["churn"].agg(["mean", "size"]).round(3))

# The three-factor profile, with baselines for each part
mtm = df.Contract == "Month-to-month"
fiber = df.InternetService == "Fiber optic"
nosec = df.OnlineSecurity == "No"
for name, mask in {"month-to-month": mtm, "fiber": fiber, "no security": nosec,
                   "all three": mtm & fiber & nosec}.items():
    print(f"{name}: n={mask.sum()}, churn={df.loc[mask, 'churn'].mean():.3f}")

