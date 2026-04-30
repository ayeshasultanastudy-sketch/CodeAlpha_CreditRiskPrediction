import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# Drop unnecessary column
df.drop("Unnamed: 0", axis=1, inplace=True)

# Handle missing values
df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
df["Checking account"] = df["Checking account"].fillna("unknown")

# Encode categorical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Housing"] = df["Housing"].map({"own": 0, "rent": 1, "free": 2})

# Create target variable (Risk)
df["Risk"] = (df["Credit amount"] > df["Credit amount"].median()).astype(int)

# Use ONLY features used in Streamlit app (IMPORTANT)
features = ["Age", "Sex", "Job", "Housing", "Credit amount", "Duration"]

X = df[features]
y = df["Risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "credit_model.pkl")

print("Model saved successfully as credit_model.pkl")