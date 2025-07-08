import pandas as pd
import streamlit as st
# Load dataset
df = pd.read_csv("C:\\Users\\vivek\\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preview
print(df.head())
print(df.info())
# 1. Drop customerID (not useful for modeling)
df.drop('customerID', axis=1, inplace=True)

# 2. Fix TotalCharges: some values are empty strings
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 3. Drop rows where TotalCharges couldn't be converted
df = df.dropna(subset=['TotalCharges'])

# 4. Ensure SeniorCitizen is categorical (0/1 → Yes/No if needed)
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

# 5. Convert target column to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 6. Reset index
df.reset_index(drop=True, inplace=True)

# ✅ Done!
print("Cleaned data shape:", df.shape)
print("Any nulls left?:\n", df.isnull().sum())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Define features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Define full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)
# Save the full pipeline to a file
joblib.dump(model_pipeline, 'churn_model.pkl')
print("✅ Model saved as churn_model.pkl")
import matplotlib.pyplot as plt
import numpy as np

# Extract trained model
rf_model = model_pipeline.named_steps['classifier']
feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
importances = rf_model.feature_importances_

# Get top 10 features
indices = np.argsort(importances)[-10:]
top_features = [feature_names[i] for i in indices]

# Plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], color='skyblue')
plt.yticks(ticks=range(len(indices)), labels=top_features)
plt.xlabel("Importance")
plt.title("Top 10 Features Influencing Churn")
plt.tight_layout()
st.pyplot(plt)

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x='Churn', hue='Churn', palette='Set2')
plt.title("Churn Distribution")
st.pyplot(plt)


