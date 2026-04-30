import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# (Same preprocessing as before)
df.drop("Unnamed: 0", axis=1, inplace=True)

df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
df["Checking account"] = df["Checking account"].fillna("unknown")

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Housing"] = df["Housing"].map({"own": 0, "rent": 1, "free": 2})

df = pd.get_dummies(df, columns=["Purpose", "Saving accounts", "Checking account"])

# Target column (IMPORTANT)
# Assuming Credit amount risk classification (we will simplify it)
df["Risk"] = (df["Credit amount"] > df["Credit amount"].median()).astype(int)

X = df.drop("Risk", axis=1)
y = df["Risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Model 2: Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Results
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\nClassification Report (Best Model):\n")
print(classification_report(y_test, rf_pred))