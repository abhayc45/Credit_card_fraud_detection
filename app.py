import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ✅ Set page config first!
st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")

# Enhanced model loading with multiple fallbacks
@st.cache_resource
def load_model():
    model_paths = [
        "credit_card_model.pkl",
        "credit_card_model.joblib", 
        "credit_card_model.json",
        "model.pkl"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                # Try joblib first
                model = joblib.load(model_path)
                st.success(f"✅ Model loaded successfully from {model_path}")
                return model
            except Exception as e1:
                try:
                    # Try regular pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    st.success(f"✅ Model loaded with pickle from {model_path}")
                    return model
                except Exception as e2:
                    st.warning(f"⚠️ Failed to load {model_path}: {str(e1)[:100]}...")
                    continue
    
    # If no model found, create a dummy model for demo
    st.error("❌ No trained model found! Using a dummy model for demonstration.")
    st.info("Please ensure your trained model file is in the same directory.")
    
    # Create a simple dummy model for demo purposes
    dummy_model = create_dummy_model()
    return dummy_model

def create_dummy_model():
    """Create a dummy model for demonstration when real model isn't available"""
    class DummyModel:
        def predict(self, X):
            # Simple rule-based prediction for demo
            predictions = []
            for _, row in X.iterrows():
                # Higher amounts and certain V features increase fraud probability
                score = 0
                if row['Amount'] > 1000:
                    score += 0.3
                if abs(row.get('V1', 0)) > 2:
                    score += 0.2
                if abs(row.get('V2', 0)) > 2:
                    score += 0.2
                if abs(row.get('V3', 0)) > 2:
                    score += 0.1
                
                predictions.append(1 if score > 0.4 else 0)
            return np.array(predictions)
        
        def predict_proba(self, X):
            # Return probability matrix
            predictions = self.predict(X)
            probs = []
            for pred in predictions:
                if pred == 1:
                    probs.append([0.2, 0.8])  # High fraud probability
                else:
                    probs.append([0.9, 0.1])  # Low fraud probability
            return np.array(probs)
    
    return DummyModel()

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Critical error loading model: {e}")
    st.stop()

# App title and description
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This application predicts whether a credit card transaction is fraudulent based on various features.
The model analyzes transaction patterns to identify potentially suspicious activities.
""")

# Sidebar with app info
with st.sidebar:
    st.header("ℹ️ App Information")
    st.markdown("""
    **Features Used:**
    - Time: Transaction timestamp
    - V1-V28: PCA transformed features
    - Amount: Transaction amount
    
    **Model Output:**
    - 0: Legitimate transaction ✅
    - 1: Fraudulent transaction ⚠️
    """)

# Dropdown to select input method
input_method = st.selectbox(
    "Choose Input Method",
    ("Manual Input", "Random Values", "Upload CSV"),
    help="Select how you want to input transaction data"
)

# Feature columns used by model
columns = ['Time'] + [f"V{i}" for i in range(1, 29)] + ['Amount']

# Create empty template
def create_empty_df():
    return pd.DataFrame(columns=columns)

# Enhanced prediction and display function
def predict_and_display(input_df):
    try:
        # Ensure all columns are present and numeric
        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0.0
        
        input_df = input_df[columns].astype(float)
        
        # Make predictions
        preds = model.predict(input_df)
        probs = model.predict_proba(input_df)[:, 1]
        
        # Add results to dataframe
        results_df = input_df.copy()
        results_df['Prediction'] = preds
        results_df['Fraud_Probability'] = probs
        results_df['Risk_Level'] = results_df['Fraud_Probability'].apply(
            lambda x: '🔴 High' if x > 0.7 else '🟡 Medium' if x > 0.3 else '🟢 Low'
        )
        
        # Display results
        st.subheader("🔍 Prediction Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(input_df))
        with col2:
            st.metric("Predicted Frauds", int(results_df['Prediction'].sum()))
        with col3:
            avg_prob = results_df['Fraud_Probability'].mean()
            st.metric("Avg Fraud Probability", f"{avg_prob:.2%}")
        with col4:
            high_risk = (results_df['Fraud_Probability'] > 0.7).sum()
            st.metric("High Risk Transactions", int(high_risk))
        
        # Detailed results table
        # Check if dataset is too large for styling
        total_cells = len(results_df) * len(results_df.columns)
        
        if total_cells > 50000:  # Limit styling for large datasets
            st.warning(f"⚠️ Large dataset detected ({len(results_df):,} rows). Showing first 1000 rows for performance.")
            
            # Show summary of full dataset
            st.write("**Full Dataset Summary:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"- Total rows: {len(results_df):,}")
                st.write(f"- Fraud transactions: {int(results_df['Prediction'].sum()):,}")
            with col2:
                st.write(f"- Fraud rate: {results_df['Prediction'].mean():.2%}")
                st.write(f"- Avg fraud probability: {results_df['Fraud_Probability'].mean():.2%}")
            
            # Show top 1000 rows with basic formatting
            display_df = results_df.head(1000).copy()
            display_df['Fraud_Probability'] = display_df['Fraud_Probability'].apply(lambda x: f"{x:.2%}")
            display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:,.2f}")
            display_df['Time'] = display_df['Time'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(display_df, use_container_width=True)
            
        else:
            # Use styling for smaller datasets
            st.dataframe(
                results_df.style.format({
                    'Fraud_Probability': '{:.2%}',
                    'Amount': '${:,.2f}',
                    'Time': '{:,.0f}'
                }),
                use_container_width=True
            )
        
        # Visualization
        if len(results_df) > 1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Prediction Distribution")
                fraud_counts = results_df['Prediction'].value_counts()
                st.bar_chart(fraud_counts)
            
            with col2:
                st.subheader("📈 Fraud Probability Distribution")
                st.bar_chart(results_df['Fraud_Probability'])
        
        # Individual transaction details (limit for large datasets)
        st.subheader("📋 Transaction Details")
        
        # Limit individual transaction display for large datasets
        max_display = min(50, len(results_df))
        if len(results_df) > max_display:
            st.info(f"Showing details for first {max_display} transactions (out of {len(results_df):,} total)")
        
        for i, (_, row) in enumerate(results_df.head(max_display).iterrows()):
            prob = row['Fraud_Probability']
            pred = row['Prediction']
            
            if pred == 1:
                st.error(f"🚨 Transaction {i+1}: **FRAUD DETECTED** (Confidence: {prob:.1%})")
            elif prob > 0.3:
                st.warning(f"⚠️ Transaction {i+1}: **SUSPICIOUS** (Confidence: {prob:.1%})")
            else:
                st.success(f"✅ Transaction {i+1}: **LEGITIMATE** (Confidence: {(1-prob):.1%})")
        
        # Download option
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Results as CSV", 
            csv, 
            "fraud_predictions.csv", 
            "text/csv",
            help="Download the predictions as a CSV file"
        )
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.write("Please check your input data format.")

# Manual Input Mode
if input_method == "Manual Input":
    st.subheader("✍️ Enter Transaction Details")
    
    with st.form("manual_input_form"):
        input_data = []
        
        # Organize inputs in a more user-friendly way
        st.write("**Basic Transaction Info:**")
        col1, col2 = st.columns(2)
        with col1:
            time_val = st.number_input("Time", value=0.0, step=1.0, help="Time in seconds from first transaction")
        with col2:
            amount_val = st.number_input("Amount ($)", value=100.0, step=0.01, help="Transaction amount in dollars")
        
        st.write("**PCA Features (V1-V28):**")
        st.caption("These are anonymized features from PCA transformation. Use values between -5 and 5.")
        
        # Create input fields for V features in a grid
        v_values = []
        for i in range(1, 29, 4):  # Group by 4s
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j <= 28:
                    with col:
                        v_val = st.number_input(
                            f"V{i+j}", 
                            value=0.0, 
                            step=0.01, 
                            format="%.4f",
                            key=f"v_{i+j}"
                        )
                        v_values.append(v_val)
        
        # Combine all inputs
        input_data = [time_val] + v_values + [amount_val]
        
        submitted = st.form_submit_button("🔎 Predict Fraud", type="primary")
        
        if submitted:
            input_df = pd.DataFrame([input_data], columns=columns)
            predict_and_display(input_df)

# Random Input Mode
elif input_method == "Random Values":
    st.subheader("🎲 Generate Random Transaction")
    st.write("This will generate a random transaction with realistic values for testing.")
    
    col1, col2 = st.columns(2)
    with col1:
        num_transactions = st.slider("Number of transactions", 1, 10, 1)
    with col2:
        fraud_bias = st.slider("Fraud bias", 0.0, 1.0, 0.1, help="Higher values increase chance of generating fraud-like patterns")
    
    if st.button("🎰 Generate and Predict", type="primary"):
        rand_data = []
        for _ in range(num_transactions):
            # Generate more realistic random data
            row = []
            
            # Time (0 to 172800 - roughly 2 days in seconds)
            time_val = np.random.uniform(0, 172800)
            row.append(time_val)
            
            # V features (PCA components, typically between -5 and 5)
            for i in range(28):
                if fraud_bias > 0.5:
                    # Introduce some patterns that might indicate fraud
                    v_val = np.random.normal(0, 2) + np.random.choice([-3, 3]) * fraud_bias
                else:
                    v_val = np.random.normal(0, 1)
                v_val = np.clip(v_val, -5, 5)
                row.append(v_val)
            
            # Amount (log-normal distribution for realistic amounts)
            if fraud_bias > 0.5:
                # Fraudulent transactions might have unusual amounts
                amount = np.random.lognormal(mean=5, sigma=2) * (1 + fraud_bias)
            else:
                amount = np.random.lognormal(mean=3, sigma=1.5)
            amount = max(0.01, min(amount, 25000))  # Cap at reasonable range
            row.append(amount)
            
            rand_data.append(row)
        
        rand_df = pd.DataFrame(rand_data, columns=columns)
        predict_and_display(rand_df)

# Upload CSV Mode
elif input_method == "Upload CSV":
    st.subheader("📁 Upload CSV File")
    st.write("Upload a CSV file containing transaction data with the required features.")
    
    # Show expected format
    with st.expander("📋 Expected CSV Format"):
        st.write("Your CSV should contain the following columns:")
        st.code(", ".join(columns))
        
        # Create sample data
        sample_data = pd.DataFrame(np.random.randn(3, len(columns)), columns=columns)
        sample_data['Amount'] = [150.0, 50.0, 1200.0]
        sample_data['Time'] = [100, 200, 300]
        st.write("**Sample format:**")
        st.dataframe(sample_data.head())
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file with transaction data"
    )
    
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Found {len(input_df)} transactions.")
            
            # Show preview
            st.write("**Data Preview:**")
            st.dataframe(input_df.head())
            
            # Check for required columns
            missing_cols = set(columns) - set(input_df.columns)
            if missing_cols:
                st.error("❌ CSV is missing required columns:")
                st.code(", ".join(sorted(missing_cols)))
                
                # Offer to fill missing columns with zeros
                if st.checkbox("Fill missing columns with zeros?"):
                    for col in missing_cols:
                        input_df[col] = 0.0
                    st.warning("⚠️ Missing columns filled with zeros. Results may be inaccurate.")
                else:
                    st.stop()
            
            # Show file size warning for large datasets
            if len(input_df) > 10000:
                st.warning(f"⚠️ Large dataset detected ({len(input_df):,} rows). Processing may take time.")
                
                # Option to sample the data
                if st.checkbox("Process sample only (recommended for testing)?"):
                    sample_size = st.slider("Sample size", 100, min(10000, len(input_df)), 1000)
                    input_df = input_df.sample(n=sample_size, random_state=42)
                    st.info(f"Using random sample of {sample_size} transactions.")
            
            # Reorder columns to match model expectations
            input_df = input_df[columns]
            
            if st.button("🔎 Predict All Transactions", type="primary"):
                with st.spinner("Processing predictions..."):
                    predict_and_display(input_df)
                
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
            st.write("Please ensure your file is a valid CSV format.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    💳 Credit Card Fraud Detection App | Built with Streamlit
</div>
""", unsafe_allow_html=True)