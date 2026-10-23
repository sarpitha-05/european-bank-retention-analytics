import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs", exist_ok=True)
sns.set_style("whitegrid")
df = pd.read_csv("bank_features.csv")
overall_churn = df["Exited"].mean()
overall_ret = 1 - overall_churn

def churn_table(by, observed=True):
    t = df.groupby(by, observed=observed)["Exited"].agg(Customers="count", Churned="sum", ChurnRate="mean")
    t["ChurnRate"] = (t["ChurnRate"] * 100).round(2)
    return t

def bar(table, title, fname, xlabel="", color="#1f4e79"):
    ax = table["ChurnRate"].plot(kind="bar", color=color, figsize=(7, 4.5), rot=0)
    ax.axhline(overall_churn * 100, color="red", ls="--", label=f"Overall {overall_churn:.1%}")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(title); ax.set_ylabel("Churn rate (%)"); ax.set_xlabel(xlabel); ax.legend()
    plt.tight_layout(); plt.savefig(f"outputs/{fname}", dpi=200); plt.close()

# =============== 1. ENGAGEMENT vs CHURN ===============
t_active = churn_table("IsActiveMember"); t_active.index = ["Inactive", "Active"]
t_profile = churn_table("EngagementProfile").loc[
    ["Active Engaged", "Active Low-Product", "Inactive Disengaged", "Inactive High-Balance"]]
t_active.to_csv("outputs/churn_by_activity.csv"); t_profile.to_csv("outputs/churn_by_profile.csv")
bar(t_active, "Churn Rate: Active vs Inactive Members", "01_churn_by_activity.png")
bar(t_profile, "Churn Rate by Engagement Profile", "02_churn_by_profile.png", color="#2e75b6")
plt.close("all")

# =============== 2. PRODUCT UTILIZATION ===============
t_prod = churn_table("NumOfProducts"); t_prod.to_csv("outputs/churn_by_products.csv")
bar(t_prod, "Churn Rate by Number of Products", "03_churn_by_products.png", "Number of products", "#c55a11")
t_tier = churn_table("ProductTier"); t_tier.to_csv("outputs/churn_by_product_tier.csv")
t_single = churn_table("MultiProduct"); t_single.index = ["Single-product", "Multi-product (2+)"]
t_single.to_csv("outputs/single_vs_multi.csv")

# product x activity heatmap
hm = df.pivot_table(index="NumOfProducts", columns="IsActiveMember", values="Exited", aggfunc="mean") * 100
hm.columns = ["Inactive", "Active"]
hm.to_csv("outputs/products_x_activity.csv")
plt.figure(figsize=(5.5, 4))
sns.heatmap(hm, annot=True, fmt=".1f", cmap="Reds", cbar_kws={"label": "Churn %"})
plt.title("Churn % : Products x Activity"); plt.tight_layout()
plt.savefig("outputs/04_products_x_activity.png", dpi=200); plt.close()

# =============== 3. FINANCIAL COMMITMENT vs ENGAGEMENT ===============
df["BalanceBand"] = pd.cut(df["Balance"], bins=[-1, 0, 50000, 100000, 150000, 300000],
                           labels=["Zero", "1-50k", "50-100k", "100-150k", "150k+"])
hm2 = df.pivot_table(index="BalanceBand", columns="IsActiveMember", values="Exited",
                     aggfunc="mean", observed=True) * 100
hm2.columns = ["Inactive", "Active"]; hm2.to_csv("outputs/balance_x_activity.csv")
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(hm2, annot=True, fmt=".1f", cmap="Reds", cbar_kws={"label": "Churn %"})
plt.title("Churn % : Balance band x Activity"); plt.tight_layout()
plt.savefig("outputs/05_balance_x_activity.png", dpi=200); plt.close()

t_risk = churn_table("AtRiskPremium"); t_risk.index = ["Other customers", "At-risk premium"]
t_risk.to_csv("outputs/at_risk_premium.csv")
t_mis = churn_table("SalaryBalanceMismatch"); t_mis.index = ["No mismatch", "Salary-balance mismatch"]
t_mis.to_csv("outputs/salary_balance_mismatch.csv")

# =============== 4. RETENTION STRENGTH ===============
t_rsi_tier = churn_table("RSITier").loc[["Weak", "Moderate", "Strong"]]
t_rsi_tier.to_csv("outputs/churn_by_rsi_tier.csv")
bar(t_rsi_tier, "Churn Rate by Relationship Strength Tier", "06_churn_by_rsi_tier.png", color="#548235")

df["RSIBand"] = pd.cut(df["RSI"], bins=range(0, 101, 10), include_lowest=True)
t_rsi = churn_table("RSIBand"); t_rsi.to_csv("outputs/churn_by_rsi_band.csv")
plt.figure(figsize=(8, 4.5))
plt.plot(range(len(t_rsi)), t_rsi["ChurnRate"], marker="o", color="#548235")
plt.xticks(range(len(t_rsi)), [str(i) for i in t_rsi.index], rotation=45, fontsize=8)
plt.axhline(overall_churn * 100, color="red", ls="--")
plt.title("Churn Rate vs Relationship Strength Index (engagement threshold)")
plt.ylabel("Churn rate (%)"); plt.tight_layout()
plt.savefig("outputs/07_churn_by_rsi_band.png", dpi=200); plt.close()

# sticky customers: active + 2 products + RSI strong
df["Sticky"] = ((df["IsActiveMember"] == 1) & (df["NumOfProducts"] == 2) & (df["RSI"] >= 70)).astype(int)
t_sticky = churn_table("Sticky"); t_sticky.index = ["Not sticky", "Sticky"]
t_sticky.to_csv("outputs/sticky_customers.csv")

# churn stability across tiers: churn by profile within each geography
stab = df.pivot_table(index="EngagementProfile", columns="Geography", values="Exited", aggfunc="mean") * 100
stab.round(1).to_csv("outputs/profile_stability_by_geography.csv")

# =============== KPIs ===============
churn_act = df.loc[df.IsActiveMember == 1, "Exited"].mean()
churn_ina = df.loc[df.IsActiveMember == 0, "Exited"].mean()
ret = lambda mask: 1 - df.loc[mask, "Exited"].mean()

kpi = {}
kpi["Engagement Retention Ratio (inactive churn / active churn)"] = round(churn_ina / churn_act, 2)
for n in sorted(df.NumOfProducts.unique()):
    kpi[f"Product Depth Index - {n} product(s) (retention / overall retention)"] = round(ret(df.NumOfProducts == n) / overall_ret, 2)
kpi["High-Balance Disengagement Rate (% of high-balance customers inactive)"] = round(df.loc[df.HighBalance == 1, "IsActiveMember"].eq(0).mean() * 100, 1)
kpi["Churn rate of at-risk premium customers (%)"] = round(df.loc[df.AtRiskPremium == 1, "Exited"].mean() * 100, 1)
kpi["Credit Card Stickiness Score (retention card / retention no card)"] = round(ret(df.HasCrCard == 1) / ret(df.HasCrCard == 0), 3)
kpi["Relationship Strength Index - mean (retained)"] = round(df.loc[df.Exited == 0, "RSI"].mean(), 1)
kpi["Relationship Strength Index - mean (churned)"] = round(df.loc[df.Exited == 1, "RSI"].mean(), 1)
kpi["Correlation RSI vs Exited"] = round(df["RSI"].corr(df["Exited"]), 3)
kpi["Sticky customer churn rate (%)"] = round(df.loc[df.Sticky == 1, "Exited"].mean() * 100, 1)
pd.Series(kpi, name="Value").to_csv("outputs/kpis.csv")

print("=== KEY TABLES ===")
for name, t in [("Active vs Inactive", t_active), ("Engagement profile", t_profile),
                ("Products", t_prod), ("Single vs multi", t_single),
                ("Products x Activity (churn %)", hm.round(1)),
                ("Balance band x Activity (churn %)", hm2.round(1)),
                ("At-risk premium", t_risk), ("Salary-balance mismatch", t_mis),
                ("RSI tier", t_rsi_tier), ("Sticky", t_sticky)]:
    print(f"\n{name}\n{t}")
print("\n=== KPIs ===")
for k, v in kpi.items():
    print(f"{k}: {v}")
print("\nSaved all tables and charts in /outputs")