import pandas as pd

df = pd.read_csv("European_Bank.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate CustomerIds:", df.CustomerId.duplicated().sum())
print("Duplicate rows:", df.duplicated().sum())

# Binary consistency
for c in ["HasCrCard", "IsActiveMember", "Exited"]:
    print(f"{c} unique values:", sorted(df[c].unique()))

# Engagement/product field validation
print("NumOfProducts values:", sorted(df.NumOfProducts.unique()))
print("Tenure range:", df.Tenure.min(), "-", df.Tenure.max())
print("Age range:", df.Age.min(), "-", df.Age.max())
print("Negative balances:", (df.Balance < 0).sum())

# Churn labeling check
print("\nChurn rate: {:.2%}".format(df.Exited.mean()))
print(df.Exited.value_counts())

# Basic cleaning: drop useless Year and Surname
df_clean = df.drop(columns=["Year", "Surname"])
df_clean.to_csv("bank_clean.csv", index=False)
print("\nSaved bank_clean.csv", df_clean.shape)