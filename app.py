"""Credit Card Fraud Detection - Streamlit App"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .fraud-alert {
        background-color: #ff4444;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .legit-alert {
        background-color: #44ff44;
        color: black;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown("Enter transaction details or upload a file to get fraud prediction")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    try:
        results_dir = "results"
        
        if not os.path.exists(results_dir):
            return None, None, None, False
        
        model_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl') and 'scaler' not in f]
        scaler_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl') and 'scaler' in f]
        
        if not model_files:
            return None, None, None, False
        
        lr_model = None
        rf_model = None
        
        for model_file in model_files:
            model_path = os.path.join(results_dir, model_file)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                if 'Logistic' in model.model_name:
                    lr_model = model
                elif 'Random' in model.model_name:
                    rf_model = model
        
        scaler = None
        if scaler_files:
            latest_scaler = sorted(scaler_files)[-1]
            scaler_path = os.path.join(results_dir, latest_scaler)
            
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        return lr_model, rf_model, scaler, True
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, False

lr_model, rf_model, scaler, models_loaded = load_models()

# Sidebar
st.sidebar.title("Settings")
st.sidebar.markdown("---")

if models_loaded:
    st.sidebar.markdown("### Choose Model")
    model_choice = st.sidebar.radio(
        "Select model:",
        ["Random Forest (Recommended)", "Logistic Regression (Faster)"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Sensitivity Level")
    
    threshold_preset = st.sidebar.select_slider(
        "Select threshold:",
        options=["Very Lenient (0.7)", "Lenient (0.6)", "Balanced (0.5)", "Strict (0.4)", "Very Strict (0.3)"],
        value="Balanced (0.5)"
    )
    
    threshold = float(threshold_preset.split("(")[1].rstrip(")"))
    
    st.sidebar.info(f"Current threshold: {threshold}")
else:
    st.sidebar.error("Models Not Found")
    st.sidebar.markdown("Please train models first using `python main.py`")

# Main content
tab1, tab2 = st.tabs(["Single Transaction", "Batch Upload"])

with tab1:
    st.header("Check a Single Transaction")
    
    if not models_loaded:
        st.error("Models not loaded. Please train models first using `python main.py`")
    else:
        st.markdown("Enter transaction amount and feature values:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Transaction Info")
            
            amount = st.number_input("💵 Amount ($)", min_value=0.0, value=100.0, step=0.01,
                                     help="Enter transaction amount")
            
            st.markdown("---")
            
            use_quick_fill = st.checkbox("✨ Quick Fill V1-V28", value=True, 
                                         help="Auto-fill features with safe values")
            
            if use_quick_fill:
                st.success("✅ Using default safe values")
            
            # Create expander for V features
            with st.expander("🔧 Advanced (Optional)" if use_quick_fill else "📊 Enter Features", expanded=not use_quick_fill):
                st.caption("These are anonymous features. Quick Fill works great for testing!")
                
                v_cols = st.columns(4)
                v_features = {}
                
                for i in range(1, 29):
                    col_idx = (i - 1) % 4
                    with v_cols[col_idx]:
                        if use_quick_fill:
                            default_val = np.random.uniform(-2, 2)  # Safe range
                        else:
                            default_val = 0.0
                        
                        v_features[f'V{i}'] = st.number_input(
                            f"V{i}", 
                            value=default_val, 
                            format="%.4f",
                            key=f"v{i}"
                        )
        
        with col2:
            st.subheader("🎯 Results")
            
            st.markdown("""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
                <h3>👈 Enter amount on the left</h3>
                <p>Click button below to check</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 CHECK FOR FRAUD", type="primary", use_container_width=True):
                # Prepare input data in correct order: V1-V28, then Amount
                input_data = pd.DataFrame({
                    **{f'V{i}': [v_features[f'V{i}']] for i in range(1, 29)},
                    'Amount': [amount]
                })
                
                # Scale amount
                input_data['Amount'] = scaler.transform(input_data[['Amount']])
                
                # Select model based on choice
                model = rf_model if "Random Forest" in model_choice else lr_model
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]
                
                # Apply threshold
                fraud_detected = probability >= threshold
                
                # Display results
                st.markdown("---")
                st.markdown("### ✅ Analysis Complete!")
                
                if fraud_detected:
                    st.markdown('<div class="fraud-alert">🚨 FRAUD DETECTED</div>', unsafe_allow_html=True)
                    st.error("⚠️ This transaction looks suspicious!")
                else:
                    st.markdown('<div class="legit-alert">✅ LEGITIMATE</div>', unsafe_allow_html=True)
                    st.success("✅ This transaction looks safe!")
                
                st.markdown("---")
                
                # Metrics
                st.markdown("### 📊 Details:")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Fraud Chance", f"{probability:.1%}")
                    st.caption("How likely it's fraud")
                
                with col_b:
                    st.metric("Your Setting", f"{threshold:.1%}")
                    st.caption("Cutoff threshold")
                
                with col_c:
                    confidence = abs(probability - 0.5) * 2
                    st.metric("Model Confidence", f"{confidence:.2%}",
                             help="How sure the model is about this prediction")
                    st.caption("How sure we are")
                
                # Probability gauge
                st.markdown("### 📊 Fraud Probability Gauge")
                fig, ax = plt.subplots(figsize=(10, 2))
                
                # Create color gradient
                colors = ['green', 'yellow', 'orange', 'red']
                cmap = plt.cm.colors.LinearSegmentedColormap.from_list('fraud', colors)
                
                # Plot bar
                ax.barh([0], [1], color='lightgray', height=0.5)
                ax.barh([0], [probability], color=cmap(probability), height=0.5)
                
                # Add threshold line
                ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
                
                # Formatting
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlabel('Probability', fontsize=12)
                ax.set_yticks([])
                ax.legend()
                ax.set_title(f'Fraud Probability: {probability:.2%}', fontsize=14, fontweight='bold')
                
                st.pyplot(fig)
                plt.close()
                
                # Additional details
                with st.expander("📋 View Transaction Details"):
                    st.json({
                        "Model Used": model_choice,
                        "Transaction Time": f"{time:.2f} seconds",
                        "Transaction Amount": f"${amount:.2f}",
                        "Fraud Probability": f"{probability:.4f}",
                        "Threshold": f"{threshold:.4f}",
                        "Classification": "FRAUD" if fraud_detected else "LEGITIMATE",
                        "Confidence Score": f"{confidence:.4f}"
                    })
                
                # Apply threshold
                fraud_detected = probability >= threshold
                
                # Display results
                st.markdown("---")
                
                if fraud_detected:
                    st.markdown('<div class="fraud-alert">⚠️ FRAUD DETECTED!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="legit-alert">✅ LEGITIMATE TRANSACTION</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Fraud Probability", f"{probability:.2%}")
                
                with col_b:
                    st.metric("Threshold", f"{threshold:.2%}")
                
                with col_c:
                    confidence = abs(probability - 0.5) * 2
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Risk assessment with clear action items
                st.markdown("### 🎯 What Should You Do?")
                if probability < 0.3:
                    st.success("""
                    ✅ **LOW RISK - Approve Transaction**
                    - This transaction looks safe
                    - Normal processing recommended
                    - No additional checks needed
                    """)
                elif probability < 0.5:
                    st.info("""
                    ⚠️ **MEDIUM RISK - Monitor Closely**
                    - Keep an eye on this transaction
                    - May want to check customer history
                    - Consider additional verification
                    """)
                elif probability < 0.7:
                    st.warning("""
                    ⚠️ **HIGH RISK - Verify Before Approving**
                    - Additional verification recommended
                    - Contact customer to confirm
                    - Review recent activity
                    """)
                else:
                    st.error("""
                    🚨 **VERY HIGH RISK - Block Immediately**
                    - Strong fraud indicators detected
                    - Block transaction immediately
                    - Contact fraud department
                    - Review customer account for compromise
                    """)

# ============================================================================
# TAB 2: BATCH ANALYSIS
# ============================================================================
with tab2:
    st.header("📊 Upload Multiple Transactions")
    
    if not models_loaded:
        st.error("❌ Models not loaded. Please run `python clean_and_run.py` first.")
    else:
        st.info("""
        ### 💡 How to use this:
        1. **Prepare a CSV file** with transaction data
        2. **Upload it** using the button below
        3. **Click Analyze** to check all transactions
        4. **Download results** with fraud predictions
        
        #### Required Columns:
        - `Amount`: Transaction amount
        - `V1` through `V28`: Transaction features
        - `Time` (optional): Transaction time
        """)
        
        uploaded_file = st.file_uploader("📁 Choose a CSV file with transactions", type=['csv'],
                                         help="Upload a CSV file with Time, Amount, and V1-V28 columns")
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} transactions")
                
                # Display sample
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10))
                
                if st.button("🚀 Analyze Batch", type="primary"):
                    with st.spinner("Analyzing transactions..."):
                        # Prepare data
                        X = df.copy()
                        if 'Class' in X.columns:
                            y_true = X.pop('Class')
                            has_labels = True
                        else:
                            has_labels = False
                        
                        # Drop Time column if it exists (model wasn't trained with it)
                        if 'Time' in X.columns:
                            X = X.drop('Time', axis=1)
                        
                        # Scale amount
                        if 'Amount' in X.columns:
                            X['Amount'] = scaler.transform(X[['Amount']])
                        
                        # Select model
                        model = rf_model if "Random Forest" in model_choice else lr_model
                        
                        # Predictions
                        predictions = model.predict(X)
                        probabilities = model.predict_proba(X)[:, 1]
                        
                        # Apply threshold
                        fraud_flags = probabilities >= threshold
                        
                        # Results
                        results_df = df.copy()
                        results_df['Fraud_Probability'] = probabilities
                        results_df['Predicted_Class'] = fraud_flags.astype(int)
                        results_df['Risk_Level'] = pd.cut(
                            probabilities, 
                            bins=[0, 0.3, 0.5, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High', 'Very High']
                        )
                        
                        # Summary
                        st.subheader("📊 Analysis Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Transactions", len(df))
                        
                        with col2:
                            fraud_count = fraud_flags.sum()
                            st.metric("Fraudulent", fraud_count, delta=f"{fraud_count/len(df)*100:.1f}%")
                        
                        with col3:
                            legit_count = len(df) - fraud_count
                            st.metric("Legitimate", legit_count, delta=f"{legit_count/len(df)*100:.1f}%")
                        
                        with col4:
                            avg_prob = probabilities.mean()
                            st.metric("Avg Fraud Prob", f"{avg_prob:.2%}")
                        
                        # Risk distribution
                        st.subheader("🎯 Risk Distribution")
                        risk_counts = results_df['Risk_Level'].value_counts()
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                        
                        # Pie chart
                        colors_pie = ['green', 'yellow', 'orange', 'red']
                        ax1.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', 
                               colors=colors_pie, startangle=90)
                        ax1.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
                        
                        # Bar chart
                        risk_counts.plot(kind='bar', ax=ax2, color=colors_pie)
                        ax2.set_title('Transaction Count by Risk Level', fontsize=14, fontweight='bold')
                        ax2.set_xlabel('Risk Level')
                        ax2.set_ylabel('Count')
                        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # If labels available, show performance
                        if has_labels:
                            from sklearn.metrics import classification_report, confusion_matrix
                            
                            st.subheader("📈 Model Performance")
                            
                            # Confusion matrix
                            cm = confusion_matrix(y_true, fraud_flags.astype(int))
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)
                            plt.close()
                            
                            # Classification report
                            st.text("Classification Report:")
                            report = classification_report(y_true, fraud_flags.astype(int))
                            st.code(report)
                        
                        # Download results
                        st.subheader("💾 Download Results")
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Analysis Results",
                            data=csv,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv"
                        )
                        
                        # Show high-risk transactions
                        st.subheader("🚨 High-Risk Transactions")
                        high_risk = results_df[results_df['Risk_Level'].isin(['High', 'Very High'])]
                        st.dataframe(high_risk.head(20))
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>💳 Credit Card Fraud Detection System | Built with Streamlit & Scikit-learn</p>
        <p>© 2026 | For Educational & Portfolio Purposes</p>
    </div>
    """,
    unsafe_allow_html=True
)
