"""
Credit Intelligence Bureau - Complete Dashboard
All pages integrated with backend API
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Credit Intelligence Bureau", layout="wide")

API_URL = "http://127.0.0.1:8000"

# Helper functions for safe conversion
def safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def format_currency(value):
    num = safe_float(value)
    return f"GHS {num:,.0f}"

def format_percentage(value):
    num = safe_float(value)
    return f"{num:.1f}%"

def format_number(value):
    num = safe_float(value)
    return f"{num:,.0f}"

# Initialize session state
if 'borrower_id' not in st.session_state:
    st.session_state.borrower_id = None
if 'borrower_name' not in st.session_state:
    st.session_state.borrower_name = None

def api_get(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        if r.status_code == 200:
            return r.json().get("data")
        return None
    except:
        return None

def api_post(endpoint, data):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=10)
        if r.status_code == 200:
            return r.json().get("data")
        return None
    except:
        return None

def check_backend():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Credit Bureau Analytics")
    st.markdown("---")
    
    backend_ok = check_backend()
    if backend_ok:
        st.success("Backend Connected")
        
        borrowers = api_get("/borrowers?limit=100")
        if borrowers:
            borrower_names = [b["full_name"] for b in borrowers]
            borrower_ids = {b["full_name"]: b["borrower_id"] for b in borrowers}
            selected = st.selectbox("Select Borrower", borrower_names)
            st.session_state.borrower_id = borrower_ids[selected]
            st.session_state.borrower_name = selected
            st.caption(f"Current: {st.session_state.borrower_name}")
    else:
        st.error("Backend Disconnected")
        st.info("Start backend: cd credit_intelligence_backend && python run.py")
    
    st.markdown("---")
    
    page = st.radio(
        "Select Dashboard View",
        [
            "Executive Summary",
            "NPL Trends",
            "Lender Performance",
            "Loan Purpose Analysis",
            "Loan Stacking Check",
            "Affordability Tool",
            "ML Model Training",
            "ML Predictions",
            "Alerts"
        ]
    )
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Header
st.markdown('<div style="font-size:2rem; color:#1E3A5F; text-align:center;">Credit Intelligence Bureau</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:1rem; color:#666; text-align:center; margin-bottom:2rem;">Ghana\'s Foremost Credit Intelligence Platform</div>', unsafe_allow_html=True)

if not backend_ok:
    st.warning("Backend is not running. Please start the backend server.")
    st.code("cd C:\\Users\\USER\\credit_intelligence_backend && python run.py")
    st.stop()

# ============================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================
if page == "Executive Summary":
    st.header("Executive Summary")
    
    if st.session_state.borrower_id:
        borrower = api_get(f"/borrowers/{st.session_state.borrower_id}")
        if borrower:
            st.info(f"Analyzing: {st.session_state.borrower_name}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Credit Score", format_number(borrower.get("credit_score", 0)))
            with col2:
                st.metric("Active Loans", format_number(borrower.get("active_loans", 0)))
            with col3:
                st.metric("Total Debt", format_currency(borrower.get("total_debt", 0)))
            with col4:
                st.metric("Monthly Income", format_currency(borrower.get("monthly_income", 0)))
    
    summary = api_get("/analytics/summary")
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Loans", format_number(summary.get("total_loans", 0)))
        with col2:
            st.metric("Default Rate", format_percentage(summary.get("default_rate", 0)))
        with col3:
            st.metric("Avg Loan Amount", format_currency(summary.get("avg_loan_amount", 0)))
        with col4:
            st.metric("Active Borrowers", format_number(summary.get("active_borrowers", 0)))
    
    st.subheader("NPL Trend")
    npl_data = api_get("/analytics/npl-trend?months=12")
    if npl_data:
        df = pd.DataFrame(npl_data)
        df['npl_ratio'] = df['npl_ratio'].apply(safe_float)
        fig = px.line(df, x='month', y='npl_ratio', markers=True, title="Monthly NPL Ratio")
        fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Target (15%)")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2: NPL TRENDS
# ============================================================
elif page == "NPL Trends":
    st.header("NPL Trends Analysis")
    npl_data = api_get("/analytics/npl-trend?months=12")
    if npl_data:
        df = pd.DataFrame(npl_data)
        df['npl_ratio'] = df['npl_ratio'].apply(safe_float)
        fig = px.bar(df, x='month', y='npl_ratio', title="Monthly NPL Ratio",
                     color='npl_ratio', color_continuous_scale='Reds', text='npl_ratio')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

# ============================================================
# PAGE 3: LENDER PERFORMANCE
# ============================================================
elif page == "Lender Performance":
    st.header("Lender Performance")
    lenders = api_get("/analytics/lender-performance")
    if lenders:
        df = pd.DataFrame(lenders)
        df['default_rate'] = df['default_rate'].apply(safe_float)
        fig = px.bar(df, x='lender_name', y='default_rate', title="Default Rate by Lender",
                     color='default_rate', color_continuous_scale='RdYlGn_r', text='default_rate')
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

# ============================================================
# PAGE 4: LOAN PURPOSE ANALYSIS
# ============================================================
elif page == "Loan Purpose Analysis":
    st.header("Loan Purpose Analysis")
    purposes = api_get("/analytics/loan-purpose")
    if purposes:
        df = pd.DataFrame(purposes)
        df['default_rate'] = df['default_rate'].apply(safe_float)
        fig = px.bar(df, x='purpose', y='default_rate', title="Default Rate by Loan Purpose",
                     color='default_rate', color_continuous_scale='Reds', text='default_rate')
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

# ============================================================
# PAGE 5: LOAN STACKING CHECK
# ============================================================
elif page == "Loan Stacking Check":
    st.header("Loan Stacking Prevention")
    st.warning("Loan stacking occurs when a borrower takes multiple loans from different lenders on the same day.")
    
    if st.session_state.borrower_id:
        if st.button("Check for Stacking", type="primary"):
            result = api_post("/stacking/check", {"borrower_id": st.session_state.borrower_id})
            if result:
                loans_today = safe_int(result.get('loans_today', 0))
                if loans_today > 0:
                    st.error(f"ALERT: Borrower has {loans_today} other loan(s) today!")
                    st.warning("Recommendation: DO NOT approve new loan today")
                else:
                    st.success("No same-day loans found")
                    st.info("Recommendation: Proceed with loan approval")
    else:
        st.info("Select a borrower from the sidebar to check for loan stacking")

# ============================================================
# PAGE 6: AFFORDABILITY TOOL
# ============================================================
elif page == "Affordability Tool":
    st.header("Affordability Calculator")
    st.info("30% Debt-to-Income Ratio is the industry-standard safe limit")
    
    if st.session_state.borrower_id:
        borrower = api_get(f"/borrowers/{st.session_state.borrower_id}")
        if borrower:
            monthly_income = safe_float(borrower.get('monthly_income', 0))
            total_debt = safe_float(borrower.get('total_debt', 0))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Monthly Income", format_currency(monthly_income))
                st.metric("Total Debt", format_currency(total_debt))
            
            max_safe = monthly_income * 0.30
            existing = total_debt / 12 if total_debt > 0 else 0
            available = max(0, max_safe - existing)
            recommended = available * 12
            
            with col2:
                st.metric("Max Safe Payment", format_currency(max_safe))
                st.metric("Existing Monthly", format_currency(existing))
                st.metric("Available Monthly", format_currency(available))
                st.metric("Recommended Loan", format_currency(recommended))
            
            if available >= 500:
                st.success("Recommendation: APPROVE")
            elif available > 0:
                st.warning("Recommendation: LIMITED APPROVAL")
            else:
                st.error("Recommendation: DECLINE")
    else:
        st.info("Select a borrower from the sidebar")

# ============================================================
# PAGE 7: ML MODEL TRAINING
# ============================================================
elif page == "ML Model Training":
    st.header("ML Model Training")
    st.markdown("---")
    
    st.info("""
    **Stacked Ensemble Architecture:**
    - Level 1: XGBoost, Random Forest, Gradient Boosting, AdaBoost
    - Level 2: Logistic Regression (Meta-Model)
    - Imbalance Handling: SMOTE-Tomek
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        st.metric("Accuracy", "80.8%")
        st.metric("Precision", "51.3%")
        st.metric("Recall", "96.7%")
    
    with col2:
        st.subheader("Model Configuration")
        st.metric("F1 Score", "0.671")
        st.metric("AUC-ROC", "89.4%")
        st.metric("Optimal Threshold", "0.258")
    
    st.markdown("---")
    st.subheader("Confusion Matrix")
    
    st.markdown("""
    | | Predicted: Paid | Predicted: Default |
    |---|---|---|
    | **Actual: Paid** | 182 (TN) | 55 (FP) |
    | **Actual: Default** | 2 (FN) | 58 (TP) |
    """)
    
    st.caption("TN: True Negatives (correctly identified paid loans)")
    st.caption("FP: False Positives (good loans flagged as risky)")
    st.caption("FN: False Negatives (defaults missed)")
    st.caption("TP: True Positives (defaults correctly identified)")
    
    st.markdown("---")
    st.subheader("Feature Importance")
    
    features = ['Credit Score', 'Past Defaults', 'Max Days Past Due', 'DTI Ratio', 'Active Loans', 'Monthly Income']
    importance = [23, 16, 11, 10, 8, 7]
    
    for f, i in zip(features, importance):
        st.progress(i / 100, text=f"{f}: {i}%")
    
    st.markdown("---")
    st.subheader("Training Data Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", "1,184")
        st.metric("Default Rate (Training)", "20.0%")
    with col2:
        st.metric("Test Samples", "297")
        st.metric("Default Rate (Test)", "20.2%")
    
    st.markdown("---")
    st.subheader("Model Interpretation")
    st.markdown("""
    - **High Recall (96.7%)**: The model catches 97% of actual defaults
    - **Moderate Precision (51.3%)**: About half of flagged cases are true defaults
    - **AUC-ROC (89.4%)**: Excellent discrimination between good and bad borrowers
    - **Optimal Threshold (0.258)**: Use this threshold for approval decisions
    """)

# ============================================================
# PAGE 8: ML PREDICTIONS
# ============================================================
elif page == "ML Predictions":
    st.header("ML Default Prediction")
    st.info("Stacked Ensemble Model - AUC: 89.4% | Recall: 96.7% | Precision: 51.3%")
    
    if st.session_state.borrower_id:
        st.subheader(f"Prediction for: {st.session_state.borrower_name}")
        
        borrower = api_get(f"/borrowers/{st.session_state.borrower_id}")
        if borrower:
            with st.expander("Borrower Features Used for Prediction"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Credit Features**")
                    st.write(f"- Credit Score: {format_number(borrower.get('credit_score', 0))}")
                    st.write(f"- Past Defaults: {format_number(borrower.get('past_defaults', 0))}")
                    st.write(f"- Active Loans: {format_number(borrower.get('active_loans', 0))}")
                    st.write(f"- Max Days Past Due: {format_number(borrower.get('max_days_past_due', 0))}")
                with col2:
                    st.write("**Financial Features**")
                    st.write(f"- Monthly Income: {format_currency(borrower.get('monthly_income', 0))}")
                    st.write(f"- Total Debt: {format_currency(borrower.get('total_debt', 0))}")
                    st.write(f"- Transaction Frequency: {format_number(borrower.get('transaction_frequency', 0))}/week")
        
        if st.button("Predict Default Risk", type="primary"):
            result = api_post("/predict", {"borrower_id": st.session_state.borrower_id})
            if result:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Default Probability", f"{safe_float(result.get('default_probability', 0))*100:.1f}%")
                    st.metric("Decision", result.get("decision", "N/A"))
                
                with col2:
                    st.metric("Risk Level", result.get("risk_level", "N/A"))
                    st.metric("Recommended Loan", format_currency(result.get("recommended_loan_amount", 0)))
                
                approval = safe_float(result.get('approval_probability', 0)) * 100
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=approval,
                    title={"text": "Approval Confidence (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2ECC71"},
                        "steps": [
                            {"range": [0, 40], "color": "#FF8B94"},
                            {"range": [40, 70], "color": "#FFD3B6"},
                            {"range": [70, 100], "color": "#A8E6CF"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                factors = result.get("factors", [])
                if factors:
                    st.subheader("Key Factors")
                    for f in factors:
                        if f.get("impact") == "positive":
                            st.success(f"Positive: {f.get('factor')}")
                        else:
                            st.error(f"Negative: {f.get('factor')}")
            else:
                st.error("Prediction failed. Make sure backend is running.")
    else:
        st.info("Select a borrower from the sidebar to make predictions")

# ============================================================
# PAGE 9: ALERTS
# ============================================================
elif page == "Alerts":
    st.header("System Alerts")
    
    lenders = api_get("/analytics/lender-performance")
    if lenders:
        high_risk = [l for l in lenders if safe_float(l.get('default_rate', 0)) > 20]
        if high_risk:
            st.error(f"CRITICAL: {len(high_risk)} lender(s) have default rates above 20%")
            for l in high_risk:
                st.write(f"  - {l['lender_name']}: {safe_float(l.get('default_rate', 0)):.1f}%")
    
    npl_data = api_get("/analytics/npl-trend?months=1")
    if npl_data and len(npl_data) > 0:
        current_npl = safe_float(npl_data[-1].get('npl_ratio', 0))
        if current_npl > 15:
            st.warning(f"Current NPL ratio is {current_npl:.1f}% - Above target of 15%")
        else:
            st.success(f"Current NPL ratio is {current_npl:.1f}% - Within target range")
    
    st.markdown("---")
    st.subheader("Model Performance Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model AUC", "89.4%", delta="Excellent")
        st.metric("Recall", "96.7%", delta="High")
    with col2:
        st.metric("Precision", "51.3%", delta="Moderate")
        st.metric("F1 Score", "0.671", delta="Acceptable")

st.markdown("---")
st.caption("Credit Intelligence Bureau - Powered by Stacked Ensemble (89.4% AUC)")