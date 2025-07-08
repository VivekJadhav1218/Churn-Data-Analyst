import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(layout="wide")

# Load model
model = joblib.load('churn_model.pkl')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\vivek\\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()

# Sidebar
st.sidebar.title("📁 Navigation")
section = st.sidebar.radio("Go to", ["📊 Data Overview", "📉 Visual Analysis", "🤖 Predict Churn"])

# Section 1: Data Overview
if section == "📊 Data Overview":
    st.title("📊 Telco Customer Churn Dataset")
    st.write("A first look at the raw data and distributions.")
    st.dataframe(df.head())

    st.markdown("### 🔍 Churn Count")
    st.write(df['Churn'].value_counts())

    st.markdown("### 🧮 Column Summary")
    st.write(df.describe(include='all'))

# Section 2: Visualizations
elif section == "📉 Visual Analysis":
    st.title("📉 Churn Visual Explorations")

    st.markdown("### 📌 Churn Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Churn', palette='Set2', ax=ax1)
    st.pyplot(fig1)

    st.markdown("### 💵 Monthly Charges by Churn")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='Set3', ax=ax2)
    st.pyplot(fig2)

    st.markdown("### ⏳ Tenure Distribution by Churn")
    fig3, ax3 = plt.subplots()
    sns.histplot(data=df, x='tenure', hue='Churn', kde=True, multiple='stack', palette='coolwarm', ax=ax3)
    st.pyplot(fig3)

# Section 3: Churn Prediction
elif section == "🤖 Predict Churn":
    st.title("🤖 Customer Churn Prediction")

    with st.form("churn_form"):
        gender = st.selectbox("Gender", ['Male', 'Female'])
        senior = st.selectbox("Senior Citizen", ['Yes', 'No'])
        partner = st.selectbox("Partner", ['Yes', 'No'])
        dependents = st.selectbox("Dependents", ['Yes', 'No'])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phoneservice = st.selectbox("Phone Service", ['Yes', 'No'])
        multiplelines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
        internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        onlinesecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        onlinebackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        deviceprotection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
        techsupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        streamingtv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
        streamingmovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
        contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        paperlessbilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
        paymentmethod = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        monthly_charges = st.number_input("Monthly Charges", value=50.0)
        totalcharges = st.number_input("Total Charges", value=100.0)

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame([{
                'gender': str(gender),
                'SeniorCitizen': 1 if senior == 'Yes' else 0,
                'Partner': str(partner),
                'Dependents': str(dependents),
                'tenure': int(tenure),
                'PhoneService': str(phoneservice),
                'MultipleLines': str(multiplelines),
                'InternetService': str(internet),
                'OnlineSecurity': str(onlinesecurity),
                'OnlineBackup': str(onlinebackup),
                'DeviceProtection': str(deviceprotection),
                'TechSupport': str(techsupport),
                'StreamingTV': str(streamingtv),
                'StreamingMovies': str(streamingmovies),
                'Contract': str(contract),
                'PaperlessBilling': str(paperlessbilling),
                'PaymentMethod': str(paymentmethod),
                'MonthlyCharges': float(monthly_charges),
                'TotalCharges': float(totalcharges)
            }])

            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]

            if pred == 1:
                st.error(f"⚠️ Customer likely to churn (Confidence: {prob:.2%})")
            else:
                st.success(f"✅ Customer likely to stay (Confidence: {1 - prob:.2%})")
