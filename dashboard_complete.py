import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Credit Intelligence Bureau", layout="wide")

API_URL = "http://127.0.0.1:8000"

# Check backend connection
def check_backend():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

# API helper functions
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

# Sidebar
with st.sidebar:
    st.title("Navigation")
    
    if check_backend():
        st.success("Backend Connected")
        backend_ok = True
    else:
        st.error("Backend Disconnected")
        backend_ok = False
    
    st.markdown("---")
    
    page = st.radio("Select Page", [
        "Executive Summary",
        "NPL Trends",
        "Lender Performance",
        "Loan Purpose Analysis",
        "Loan Stacking Check",
        "Affordability Tool",
        "ML Model Training",
        "ML Predictions",
        "Alerts"
    ])

st.title("Credit Intelligence Bureau")
st.markdown("---")

if not backend_ok:
    st.warning("Backend is not running. Start it with: cd credit_intelligence_backend && python run.py")
    st.stop()

# ============================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================
if page == "Executive Summary":
    st.header("Executive Summary")
    
    summary = api_get("/analytics/summary")
    
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Loans", summary.get("total_loans", 0))
        col2.metric("Total Borrowers", summary.get("total_borrowers", 0))
        col3.metric("Total Lenders", summary.get("total_lenders", 0))
        col4.metric("Default Rate", f"{summary.get('default_rate', 0)}%")
    
    st.subheader("NPL Trend")
    npl_data = api_get("/analytics/npl-trend?months=12")
    if npl_data:
        df = pd.DataFrame(npl_data)
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
        
        fig = px.bar(df, x='month', y='npl_ratio', 
                     title="Monthly NPL Ratio",
                     color='npl_ratio',
                     color_continuous_scale='Reds',
                     text='npl_ratio')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Monthly Data")
        st.dataframe(df, use_container_width=True)

# ============================================================
# PAGE 3: LENDER PERFORMANCE
# ============================================================
elif page == "Lender Performance":
    st.header("Lender Performance")
    
    lenders = api_get("/analytics/lender-performance")
    if lenders:
        df = pd.DataFrame(lenders)
        
        fig = px.bar(df, x='lender_name', y='default_rate', 
                     title="Default Rate by Lender",
                     color='default_rate',
                     color_continuous_scale='RdYlGn_r',
                     text='default_rate')
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Lender Rankings")
        st.dataframe(df, use_container_width=True)

# ============================================================
# PAGE 4: LOAN PURPOSE ANALYSIS
# ============================================================
elif page == "Loan Purpose Analysis":
    st.header("Loan Purpose Analysis")
    
    purposes = api_get("/analytics/loan-purpose")
    if purposes:
        df = pd.DataFrame(purposes)
        
        fig = px.bar(df, x='purpose', y='default_rate', 
                     title="Default Rate by Loan Purpose",
                     color='default_rate',
                     color_continuous_scale='Reds',
                     text='default_rate')
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
    
    borrowers = api_get("/borrowers?limit=100")
    if borrowers:
        borrower_dict = {f"{b['full_name']} (ID: {b['borrower_id']})": b['borrower_id'] for b in borrowers}
        selected = st.selectbox("Select Borrower", list(borrower_dict.keys()))
        
        if st.button("Check for Stacking", type="primary"):
            result = api_post("/stacking/check", {"borrower_id": borrower_dict[selected]})
            if result:
                loans_today = result.get('loans_today', 0)
                if loans_today > 0:
                    st.error(f"ALERT: Borrower has {loans_today} other loan(s) today!")
                    st.warning("Recommendation: DO NOT approve new loan today")
                else:
                    st.success("No same-day loans found")
                    st.info("Recommendation: Proceed with loan approval")

# ============================================================
# PAGE 6: AFFORDABILITY TOOL WITH TRAINED ML MODEL
# ============================================================
elif page == "Affordability Tool":
    st.header("Affordability Calculator with Trained ML Model")
    st.info("Combines 30% DTI Rule with Stacked Ensemble ML Model (89.4% AUC)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Borrower Financial Input")
        income = st.number_input("Monthly Income (GHS)", min_value=0, value=3000, step=500)
        debt = st.number_input("Total Existing Debt (GHS)", min_value=0, value=2000, step=500)
    
    if st.button("Calculate Risk Assessment", type="primary"):
        result = api_post("/affordability/calculate", {"monthly_income": income, "total_debt": debt})
        
        if result:
            # DTI Results
            with col2:
                st.subheader("DTI-Based Assessment")
                dti_data = result.get("dti", {})
                
                st.metric("Max Safe Payment (30%)", f"GHS {dti_data.get('max_safe_payment', 0):,.0f}")
                st.metric("Existing Monthly Debt", f"GHS {dti_data.get('existing_monthly', 0):,.0f}")
                st.metric("Available Monthly", f"GHS {dti_data.get('available_monthly', 0):,.0f}")
                st.metric("Recommended Loan (12mo)", f"GHS {dti_data.get('recommended_loan', 0):,.0f}")
                
                dti_decision = dti_data.get('decision', '')
                if dti_decision == "APPROVE":
                    st.success(f"DTI: {dti_decision}")
                elif dti_decision == "LIMITED":
                    st.warning(f"DTI: {dti_decision}")
                else:
                    st.error(f"DTI: {dti_decision}")
            
            # ML Results
            st.subheader("Trained ML Model Assessment")
            st.caption("Stacked Ensemble - XGBoost, Random Forest, GBM, AdaBoost")
            
            ml_data = result.get("ml", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Default Probability", f"{ml_data.get('default_probability', 0)*100:.1f}%")
                st.metric("ML Decision", ml_data.get("decision", "N/A"))
                st.metric("Risk Level", ml_data.get("risk_level", "N/A"))
            with col2:
                st.metric("Approval Confidence", f"{ml_data.get('approval_probability', 0)*100:.1f}%")
                st.metric("ML Recommended Loan", f"GHS {ml_data.get('recommended_loan', 0):,.0f}")
            
            # ML Confidence Gauge
            approval = ml_data.get('approval_probability', 0) * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=approval,
                title={"text": "ML Approval Confidence (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2ECC71"},
                    "steps": [
                        {"range": [0, 40], "color": "#FF8B94"},
                        {"range": [40, 70], "color": "#FFD3B6"},
                        {"range": [70, 100], "color": "#A8E6CF"}
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "value": 70}
                }
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
            
            # Combined Final Decision
            st.subheader("Combined Recommendation")
            final_data = result.get("final", {})
            final_decision = final_data.get("decision", "N/A")
            reasoning = final_data.get("reasoning", "")
            
            if final_decision == "APPROVE":
                st.success(f"FINAL: {final_decision}")
            elif final_decision == "DECLINE":
                st.error(f"FINAL: {final_decision}")
            else:
                st.warning(f"FINAL: {final_decision}")
            
            st.info(f"Reasoning: {reasoning}")
            
            # Risk Factors
            risk_factors = final_data.get("risk_factors", [])
            if risk_factors:
                st.subheader("Key Risk Factors")
                for factor in risk_factors:
                    if factor.get("impact") == "positive":
                        st.success(f"Positive: {factor.get('factor')}")
                    else:
                        st.error(f"Negative: {factor.get('factor')}")

# ============================================================
# PAGE 7: ML MODEL TRAINING
# ============================================================
elif page == "ML Model Training":
    st.header("ML Model Training")
    st.info("""
    **Stacked Ensemble Architecture:**
    - Level 1: XGBoost, Random Forest, Gradient Boosting, AdaBoost
    - Level 2: Logistic Regression (Meta-Model)
    - Imbalance Handling: SMOTE-Tomek
    - AUC-ROC: 89.4% | Recall: 96.7% | Precision: 51.3%
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
    
    st.subheader("Feature Importance")
    features = ['Credit Score', 'Past Defaults', 'Max Days Past Due', 'DTI Ratio', 'Active Loans', 'Monthly Income']
    importance = [23, 16, 11, 10, 8, 7]
    for f, i in zip(features, importance):
        st.progress(i / 100, text=f"{f}: {i}%")

# ============================================================
# PAGE 8: ML PREDICTIONS
# ============================================================
elif page == "ML Predictions":
    st.header("ML Default Prediction")
    st.info("Stacked Ensemble Model - AUC: 89.4% | Recall: 96.7%")
    
    borrowers = api_get("/borrowers?limit=100")
    if borrowers:
        borrower_dict = {b["full_name"]: b["borrower_id"] for b in borrowers}
        selected = st.selectbox("Select Borrower", list(borrower_dict.keys()))
        
        if st.button("Predict Default Risk", type="primary"):
            result = api_post("/predict", {"borrower_id": borrower_dict[selected]})
            if result:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Default Probability", f"{result.get('default_probability', 0)*100:.1f}%")
                    st.metric("Decision", result.get("decision", "N/A"))
                with col2:
                    st.metric("Risk Level", result.get("risk_level", "N/A"))
                    st.metric("Recommended Loan", f"GHS {result.get('recommended_loan_amount', 0):,.0f}")
                
                approval = result.get('approval_probability', 0) * 100
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
        st.warning("No borrowers found")

# ============================================================
# PAGE 9: ALERTS
# ============================================================
elif page == "Alerts":
    st.header("System Alerts and Recommendations")
    
    # Lender alerts
    lenders = api_get("/analytics/lender-performance")
    if lenders:
        high_risk = [l for l in lenders if float(l.get('default_rate', 0)) > 20]
        if high_risk:
            st.error(f"CRITICAL: {len(high_risk)} lender(s) have default rates above 20%")
            for l in high_risk:
                st.write(f"  - {l['lender_name']}: {l['default_rate']}% default rate")
    
    # Loan purpose alerts
    purposes = api_get("/analytics/loan-purpose")
    if purposes:
        high_risk_purposes = [p for p in purposes if float(p.get('default_rate', 0)) > 25]
        if high_risk_purposes:
            st.warning(f"WARNING: {len(high_risk_purposes)} loan purpose(s) have default rates above 25%")
            for p in high_risk_purposes:
                st.write(f"  - {p['purpose']}: {p['default_rate']}% default rate")
    
    # NPL alert
    npl_data = api_get("/analytics/npl-trend?months=1")
    if npl_data and len(npl_data) > 0:
        try:
            current_npl = float(npl_data[-1].get('npl_ratio', 0))
            if current_npl > 15:
                st.warning(f"Current NPL ratio is {current_npl}% - Above target of 15%")
            else:
                st.success(f"Current NPL ratio is {current_npl}% - Within target range")
        except:
            pass
    
    st.markdown("---")
    st.subheader("Recommended Actions")
    st.markdown("""
    1. Monitor NPL ratio daily
    2. Schedule risk reviews with high-default lenders
    3. Tighten underwriting for emergency loans
    4. Continue real-time loan stacking prevention
    """)

st.markdown("---")
st.caption("Credit Intelligence Bureau - Powered by Stacked Ensemble (89.4% AUC)")