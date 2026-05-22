"""
XDSData Ghana - Management Dashboard
Now connects to FastAPI backend instead of direct database
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests

# Page configuration
st.set_page_config(
    page_title="Credit Intelligence Bureau Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


def check_backend():
    """Check if backend API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False


def api_get(endpoint):
    """Make GET request to backend"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        if response.status_code == 200:
            return response.json().get("data")
        return None
    except:
        return None


def api_post(endpoint, data):
    """Make POST request to backend"""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=10)
        if response.status_code == 200:
            return response.json().get("data")
        return None
    except:
        return None


def main():
    # Header
    st.markdown('<div class="main-header">🏦 Credit Intelligence Bureau</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ghana\'s Foremost Credit Intelligence Platform</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        
        # Backend status
        if check_backend():
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Disconnected - Run 'python run.py' in backend folder")
        
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
                "ML Predictions"
            ]
        )
        
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Route to pages
    if page == "Executive Summary":
        show_executive_summary()
    elif page == "NPL Trends":
        show_npl_trends()
    elif page == "Lender Performance":
        show_lender_performance()
    elif page == "Loan Purpose Analysis":
        show_loan_purpose()
    elif page == "Loan Stacking Check":
        show_loan_stacking()
    elif page == "Affordability Tool":
        show_affordability_tool()
    elif page == "ML Predictions":
        show_ml_predictions()


def show_executive_summary():
    st.header("Executive Summary")
    st.markdown("---")
    
    summary = api_get("/analytics/summary")
    
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Loans", summary.get('total_loans', 0))
        with col2:
            st.metric("Total Borrowers", summary.get('total_borrowers', 0))
        with col3:
            st.metric("Default Rate", f"{summary.get('default_rate', 0)}%")
        with col4:
            st.metric("Avg Loan Amount", f"GHS {summary.get('avg_loan_amount', 0):,.0f}")
    else:
        st.warning("Could not load dashboard summary")
    
    # NPL Trend
    st.subheader("NPL Ratio Trend")
    npl_data = api_get("/analytics/npl-trend?months=12")
    
    if npl_data:
        df = pd.DataFrame(npl_data)
        fig = px.line(df, x='month', y='npl_ratio', markers=True, title="Monthly NPL Trend")
        fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Target (15%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No NPL trend data available")


def show_npl_trends():
    st.header("NPL Trends Analysis")
    st.markdown("---")
    
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
    else:
        st.info("No NPL trend data available")


def show_lender_performance():
    st.header("Lender Performance")
    st.markdown("---")
    
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
        
        st.subheader("Detailed Data")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No lender performance data available")


def show_loan_purpose():
    st.header("Loan Purpose Analysis")
    st.markdown("---")
    
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
        
        # Highlight high-risk purposes
        high_risk = df[df['default_rate'] > 25]
        if not high_risk.empty:
            st.warning(f"⚠️ High Risk Purposes: {', '.join(high_risk['purpose'].tolist())}")
    else:
        st.info("No loan purpose data available")


def show_loan_stacking():
    st.header("Loan Stacking Prevention")
    st.markdown("---")
    
    st.warning("""
    **What is Loan Stacking?**
    
    Loan stacking occurs when a borrower takes multiple loans from different lenders on the same day.
    This dashboard checks for same-day loans to prevent over-indebtedness.
    """)
    
    st.markdown("---")
    st.subheader("Check Borrower")
    
    borrowers = api_get("/borrowers?limit=100")
    
    if borrowers:
        borrower_options = {f"{b['full_name']} (ID: {b['borrower_id']})": b['borrower_id'] 
                           for b in borrowers}
        selected_display = st.selectbox("Select Borrower", list(borrower_options.keys()))
        selected_id = borrower_options[selected_display]
        
        if st.button("Check for Stacking", type="primary"):
            result = api_post("/stacking/check", {"borrower_id": selected_id})
            
            if result:
                loans_today = result.get('loans_today', 0)
                
                if loans_today > 0:
                    st.error(f"🚨 ALERT: Borrower has {loans_today} other loan(s) today!")
                    st.markdown("**Recommendation:** DO NOT approve new loan today")
                else:
                    st.success(f"✅ No same-day loans found")
                    st.markdown("**Recommendation:** Proceed with loan approval")
    else:
        st.warning("No borrowers found")


def show_affordability_tool():
    st.header("Affordability Calculator")
    st.markdown("---")
    
    st.info("""
    **How this tool works:**
    
    1. Mobile Money History estimates monthly income
    2. Existing loan obligations are current debt payments
    3. 30% Debt-to-Income Ratio is the industry-standard safe limit
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Monthly Income (GHS)", min_value=0, value=3000, step=500)
        debt = st.number_input("Total Existing Debt (GHS)", min_value=0, value=2000, step=500)
    
    if st.button("Calculate Affordability", type="primary"):
        result = api_post("/affordability/calculate", {"monthly_income": income, "total_debt": debt})
        
        if result:
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result.get('available_monthly', 0),
                    title={"text": "Available Monthly (GHS)"},
                    gauge={
                        "axis": {"range": [0, result.get('max_safe_payment', 1000)]},
                        "bar": {"color": "#2ECC71"}
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Results:**
            - Max Safe Payment (30%): **GHS {result.get('max_safe_payment', 0):,.0f}**
            - Existing Monthly Debt: **GHS {result.get('existing_monthly', 0):,.0f}**
            - Available Monthly: **GHS {result.get('available_monthly', 0):,.0f}**
            - Recommended Loan (12 months): **GHS {result.get('recommended_loan', 0):,.0f}**
            """)
            
            decision = result.get('decision', '')
            if decision == "APPROVE":
                st.success("✅ Recommendation: APPROVE")
            elif decision == "LIMITED":
                st.warning("⚠️ Recommendation: LIMITED APPROVAL")
            else:
                st.error("❌ Recommendation: DECLINE")


def show_ml_predictions():
    st.header("ML Default Prediction")
    st.markdown("---")
    
    st.info("""
    **Stacked Ensemble Model Performance:**  
    - AUC-ROC: 89.4% | Recall: 96.7% | Precision: 51.3%
    - Base Models: XGBoost, Random Forest, Gradient Boosting, AdaBoost
    - Meta-Model: Logistic Regression
    """)
    
    borrowers = api_get("/borrowers?limit=100")
    
    if borrowers:
        borrower_options = {f"{b['full_name']} (Score: {b.get('credit_score', 'N/A')})": b['borrower_id'] 
                           for b in borrowers}
        selected_display = st.selectbox("Select Borrower", list(borrower_options.keys()))
        selected_id = borrower_options[selected_display]
        
        if st.button("Predict Default Risk", type="primary"):
            with st.spinner("Calling ML model..."):
                prediction = api_post("/predict", {"borrower_id": selected_id})
            
            if prediction:
                col1, col2 = st.columns(2)
                
                with col1:
                    approval_pct = prediction.get('approval_probability', 0) * 100
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=approval_pct,
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
                
                with col2:
                    st.metric("Default Probability", f"{prediction.get('default_probability', 0)*100:.1f}%")
                    st.metric("Decision", prediction.get('decision', 'N/A'))
                    st.metric("Risk Level", prediction.get('risk_level', 'N/A'))
                    st.metric("Recommended Loan", f"GHS {prediction.get('recommended_loan_amount', 0):,.0f}")
                
                # Factors
                factors = prediction.get('factors', [])
                if factors:
                    st.subheader("Key Factors")
                    for factor in factors:
                        if factor.get('impact') == 'positive':
                            st.success(f"✅ {factor.get('factor')}")
                        else:
                            st.error(f"❌ {factor.get('factor')}")
    else:
        st.warning("No borrowers found")


if __name__ == "__main__":
    main()