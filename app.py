import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Set page config first!
st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("credit_card_model.pkl")

model = load_model()

st.title("💳 Credit Card Fraud Detection")
# ... rest of your app ...


# Dropdown to select input method
input_method = st.selectbox(
    "Choose Input Method",
    ("Manual Input", "Random Values", "Upload CSV")
)

# Feature columns used by model (REMOVED 'Hour')
columns = ['Time'] + [f"V{i}" for i in range(1, 29)] + ['Amount']

# Create empty template
def create_empty_df():
    return pd.DataFrame(columns=columns)

# Predict and display results
def predict_and_display(input_df):
    try:
        input_df = input_df[columns].astype(float)
    except Exception as e:
        st.error(f"❌ Data type error: {e}")
        st.stop()

    preds = model.predict(input_df)
    probs = model.predict_proba(input_df)[:, 1]
    input_df['Prediction'] = preds
    input_df['Fraud_Probability'] = probs

    st.subheader("🔍 Prediction Results")
    st.dataframe(input_df)

    st.metric("Total Transactions", len(input_df))
    st.metric("Predicted Frauds", input_df['Prediction'].sum())
    st.bar_chart(input_df['Prediction'].value_counts())

    csv = input_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results as CSV", csv, "fraud_predictions.csv", "text/csv")

    for i, prob in enumerate(input_df['Fraud_Probability']):
        st.write(f"Transaction {i+1}: {'⚠️ Fraud' if prob > 0.5 else '✅ Legitimate'} (Confidence: {prob:.2f})")

# Manual Input Mode
if input_method == "Manual Input":
    st.subheader("✍️ Enter Transaction Details")
    input_data = []
    col1, col2 = st.columns(2)

    with col1:
        for col in columns[:len(columns)//2]:
            input_data.append(st.number_input(col, value=0.0, step=0.01, format="%.4f"))

    with col2:
        for col in columns[len(columns)//2:]:
            input_data.append(st.number_input(col, value=0.0, step=0.01, format="%.4f"))

    if st.button("🔎 Predict Fraud"):
        input_df = pd.DataFrame([input_data], columns=columns)
        predict_and_display(input_df)

# Random Input Mode
elif input_method == "Random Values":
    st.subheader("🎲 Generate Random Transaction")
    if st.button("🎰 Generate and Predict"):
        rand_data = np.clip(np.random.normal(0, 1, size=(1, len(columns))), -5, 5)
        rand_df = pd.DataFrame(rand_data, columns=columns)
        rand_df['Amount'] = np.abs(rand_df['Amount']) * 1000
        rand_df['Time'] = np.abs(rand_df['Time']) * 100000
        predict_and_display(rand_df)

# Upload CSV Mode
elif input_method == "Upload CSV":
    st.subheader("📁 Upload CSV File with Required Features")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"📄 Failed to read CSV: {e}")
            st.stop()

        missing_cols = set(columns) - set(input_df.columns)
        if missing_cols:
            st.error("❌ CSV is missing required columns:")
            st.code(", ".join(missing_cols), language="text")
            st.stop()

        input_df = input_df[columns]  # Reorder correctly
        predict_and_display(input_df)
