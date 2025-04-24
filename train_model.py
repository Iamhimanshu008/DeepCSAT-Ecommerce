# train_model.py


import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/eCommerce_Customer_support_data.csv")

# Use subset of useful columns
cols = [
    'channel_name', 'category', 'Sub-category',
    'Agent_name', 'Supervisor', 'Manager',
    'Tenure Bucket', 'Agent Shift', 'CSAT Score'
]
df = df[cols].dropna()

# Binary target
df['CSAT Score'] = df['CSAT Score'].apply(lambda x: 1 if x >= 4 else 0)

# Sample
df = df.sample(n=5000, random_state=42)

X = df.drop('CSAT Score', axis=1)
y = df['CSAT Score']

# Preprocessing
cat_cols = X.columns.tolist()
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
])

# Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, pipeline.predict(X_test))
print(f"✅ Test Accuracy: {accuracy:.2f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline.named_steps['clf'], "model/model.pkl")
joblib.dump(pipeline.named_steps['pre'], "model/preprocessor.pkl")
print("✅ Model and preprocessor saved in /model/")
