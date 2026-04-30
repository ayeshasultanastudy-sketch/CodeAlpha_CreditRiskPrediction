import pandas as pd

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# Show first 5 rows
print(df.head())

# Show dataset info
print(df.info())

# Show missing values
print(df.isnull().sum())