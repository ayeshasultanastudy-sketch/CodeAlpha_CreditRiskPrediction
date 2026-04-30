import pandas as pd

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# 1. Drop useless column
df.drop("Unnamed: 0", axis=1, inplace=True)

# 2. Fill missing values
df["Saving accounts"].fillna("unknown", inplace=True)
df["Checking account"].fillna("unknown", inplace=True)

# 3. Encode categorical columns
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Housing"] = df["Housing"].map({"own": 0, "rent": 1, "free": 2})

# 4. One-hot encode remaining categorical columns
df = pd.get_dummies(df, columns=["Purpose", "Saving accounts", "Checking account"])

# 5. Show cleaned data
print(df.head())

# 6. Check final shape
print("Final shape:", df.shape)