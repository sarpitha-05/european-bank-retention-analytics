import pandas as pd
import numpy as np

df = pd.read_csv("bank_clean.csv")

# ---------- Thresholds (documented in the paper) ----------
HIGH_BAL = df["Balance"].quantile(0.75)          # top 25% balance = "high balance"
MED_SAL = df["EstimatedSalary"].median()
print(f"High-balance threshold (75th pct): {HIGH_BAL:,.0f}")
print(f"Median salary: {MED_SAL:,.0f}")

# ---------- Basic flags ----------
df["HighBalance"] = (df["Balance"] >= HIGH_BAL).astype(int)
df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
df["MultiProduct"] = (df["NumOfProducts"] >= 2).astype(int)

# ---------- Engagement profiles (mutually exclusive, priority order) ----------
def profile(r):
    if r.IsActiveMember == 0 and r.HighBalance == 1:
        return "Inactive High-Balance"
    if r.IsActiveMember == 0:
        return "Inactive Disengaged"
    if r.NumOfProducts == 1:
        return "Active Low-Product"
    return "Active Engaged"

df["EngagementProfile"] = df.apply(profile, axis=1)

# ---------- Product tier ----------
df["ProductTier"] = pd.cut(df["NumOfProducts"], bins=[0, 1, 2, 4],
                           labels=["1 Product", "2 Products", "3-4 Products"])

# ---------- Salary-balance mismatch ----------
df["BalanceSalaryRatio"] = df["Balance"] / df["EstimatedSalary"]
# high earner who keeps (almost) nothing with the bank
df["SalaryBalanceMismatch"] = ((df["EstimatedSalary"] >= MED_SAL) &
                               (df["BalanceSalaryRatio"] < 0.10)).astype(int)

# ---------- At-risk premium customers ----------
df["AtRiskPremium"] = ((df["HighBalance"] == 1) &
                       (df["IsActiveMember"] == 0)).astype(int)

# ---------- Demographic bands (for filters / context only) ----------
df["AgeGroup"] = pd.cut(df["Age"], bins=[17, 30, 40, 50, 60, 100],
                        labels=["18-30", "31-40", "41-50", "51-60", "60+"])
df["TenureGroup"] = pd.cut(df["Tenure"], bins=[-1, 2, 5, 8, 10],
                           labels=["0-2 yrs", "3-5 yrs", "6-8 yrs", "9-10 yrs"])
df["CreditBand"] = pd.cut(df["CreditScore"], bins=[0, 579, 669, 739, 850],
                          labels=["Poor", "Fair", "Good", "Excellent"])

# ---------- Relationship Strength Index (0-100) ----------
# Active member 40 | product depth 25 (1 product=12.5, 2+=25) |
# tenure 15 | credit card 10 | holds a balance 10
df["RSI"] = (df["IsActiveMember"] * 40
             + np.minimum(df["NumOfProducts"], 2) / 2 * 25
             + df["Tenure"] / 10 * 15
             + df["HasCrCard"] * 10
             + (1 - df["ZeroBalance"]) * 10)
df["RSI"] = df["RSI"].round(1)
df["RSITier"] = pd.cut(df["RSI"], bins=[-1, 40, 70, 100],
                       labels=["Weak", "Moderate", "Strong"])

df.to_csv("bank_features.csv", index=False)
print("\nSaved bank_features.csv", df.shape)

# ---------- Quick sanity checks ----------
print("\nChurn by Engagement Profile:")
print(df.groupby("EngagementProfile")["Exited"].agg(["count", "mean"]).round(3))
print("\nChurn by RSI tier:")
print(df.groupby("RSITier", observed=True)["Exited"].agg(["count", "mean"]).round(3))
print("\nAt-risk premium customers:", df["AtRiskPremium"].sum(),
      "| churn:", round(df.loc[df.AtRiskPremium == 1, "Exited"].mean(), 3))
print("Salary-balance mismatch customers:", df["SalaryBalanceMismatch"].sum(),
      "| churn:", round(df.loc[df.SalaryBalanceMismatch == 1, "Exited"].mean(), 3))