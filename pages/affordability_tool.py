"""
Affordability Tool Page with Complete ML Model Integration
Automatically uses the selected borrower from sidebar session state
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from utils.session_state import SessionState

API_URL = "http://127.0.0.1:8000"

def api_post(endpoint, data):
    """Make POST request to backend"""
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=10)
        if r.status_code == 200:
            return r.json().get("data")
        return None
    except:
        return None

def show(db=None):
    """Main function for affordability tool page"""
    
    st.header("Affordability Calculator with Trained ML Model")
    st.markdown("---")
    
    st.info("""
    **How this tool works:**
    
    Combines two assessment methods:
    1. **DTI Rule (30%)** - Traditional debt-to-income calculation
    2. **Stacked Ensemble ML Model** - Trained on 2,000+ loans (89.4% AUC)
    """)
    
    # Get current borrower from session state
    current_borrower_id = SessionState.get_borrower_id()
    current_borrower_name = SessionState.get_borrower_name()
    current_borrower_data = SessionState.get_borrower_data()
    
    # ============================================================
    # Check if borrower is selected
    # ============================================================
    if not current_borrower_id or not current_borrower_data:
        st.warning("⚠️ No borrower selected. Please select a borrower from the sidebar dropdown.")
        st.info("Go to the sidebar and select a borrower to analyze their affordability.")
        return
    
    # ============================================================
    # Display Selected Borrower Information
    # ============================================================
    st.subheader(f"📋 Borrower Analysis: {current_borrower_name}")
    st.markdown(f"**Borrower ID:** {current_borrower_id}")
    
    # Display key borrower metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        credit_score = current_borrower_data.get('credit_score', 'N/A')
        if credit_score != 'N/A':
            credit_score = int(credit_score)
        st.metric("Credit Score", credit_score)
    
    with col2:
        st.metric("Active Loans", current_borrower_data.get('active_loans', 0))
    
    with col3:
        total_debt = current_borrower_data.get('total_debt', 0)
        st.metric("Total Debt", f"GHS {total_debt:,.0f}")
    
    with col4:
        monthly_income = current_borrower_data.get('monthly_income', 0)
        st.metric("Monthly Income", f"GHS {monthly_income:,.0f}")
    
    with col5:
        past_defaults = current_borrower_data.get('past_defaults', 0)
        st.metric("Past Defaults", past_defaults)
    
    st.markdown("---")
    
    # ============================================================
    # Calculate Affordability Button
    # ============================================================
    if st.button("Calculate Risk Assessment", type="primary", use_container_width=True):
        with st.spinner(f"Calculating affordability for {current_borrower_name}..."):
            
            # Prepare request data from selected borrower
            request_data = {
                "monthly_income": float(current_borrower_data.get('monthly_income', 0)),
                "total_debt": float(current_borrower_data.get('total_debt', 0)),
                "credit_score": float(current_borrower_data.get('credit_score', 650)),
                "past_defaults": float(current_borrower_data.get('past_defaults', 0)),
                "max_days_past_due": float(current_borrower_data.get('max_days_past_due', 0)),
                "active_loans": float(current_borrower_data.get('active_loans', 0)),
                "transaction_frequency": float(current_borrower_data.get('transaction_frequency', 10)),
                "airtime_consistency": float(current_borrower_data.get('airtime_consistency', 0.7)),
                "night_applications": float(current_borrower_data.get('night_applications', 0)),
                "utility_score": float(current_borrower_data.get('utility_score', 0.7)),
                "age": float(current_borrower_data.get('age', 35)),
                "customer_tenure_days": float(current_borrower_data.get('customer_tenure_days', 365))
            }
            
            result = api_post("/affordability/calculate", request_data)
            
            if result:
                display_results(result, current_borrower_name)
            else:
                st.error("Failed to get response from backend. Make sure API is running.")


def display_results(result, borrower_name):
    """Display calculation results with borrower context"""
    
    if not result:
        st.error("No results to display")
        return
    
    st.subheader(f"📊 Assessment Results for {borrower_name}")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    # ============================================================
    # DTI Results
    # ============================================================
    with col1:
        st.subheader("💰 DTI-Based Assessment")
        st.caption("30% Debt-to-Income Rule")
        
        dti_data = result.get("dti", {})
        
        st.metric(
            "Max Safe Payment (30%)", 
            f"GHS {dti_data.get('max_safe_payment', 0):,.0f}"
        )
        st.metric(
            "Existing Monthly Debt", 
            f"GHS {dti_data.get('existing_monthly', 0):,.0f}"
        )
        st.metric(
            "Available Monthly", 
            f"GHS {dti_data.get('available_monthly', 0):,.0f}"
        )
        st.metric(
            "Recommended Loan (12mo)", 
            f"GHS {dti_data.get('recommended_loan', 0):,.0f}"
        )
        
        dti_decision = dti_data.get('decision', '')
        if dti_decision == "APPROVE":
            st.success(f"✅ DTI Decision: {dti_decision}")
        elif dti_decision == "LIMITED":
            st.warning(f"⚠️ DTI Decision: {dti_decision}")
        else:
            st.error(f"❌ DTI Decision: {dti_decision}")
    
    # ============================================================
    # ML Results
    # ============================================================
    with col2:
        st.subheader("🤖 Trained ML Model Assessment")
        st.caption("Stacked Ensemble - XGBoost, RF, GBM, AdaBoost (89.4% AUC)")
        
        ml_data = result.get("ml", {})
        
        # Risk gauge
        default_prob = ml_data.get('default_probability', 0) * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=default_prob,
            title={"text": f"Default Probability: {default_prob:.1f}%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#E74C3C" if default_prob > 50 else "#F39C12" if default_prob > 30 else "#2ECC71"},
                "steps": [
                    {"range": [0, 30], "color": "#A8E6CF", "name": "Low Risk"},
                    {"range": [30, 50], "color": "#FFD3B6", "name": "Medium Risk"},
                    {"range": [50, 100], "color": "#FF8B94", "name": "High Risk"}
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 50}
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        col_a, col_b = st.columns(2)
        with col_a:
            ml_decision = ml_data.get("decision", "N/A")
            if ml_decision == "APPROVE":
                st.success(f"ML Decision: {ml_decision}")
            elif ml_decision == "DECLINE":
                st.error(f"ML Decision: {ml_decision}")
            else:
                st.warning(f"ML Decision: {ml_decision}")
            st.metric("Risk Level", ml_data.get("risk_level", "N/A"))
        with col_b:
            st.metric("Approval Confidence", f"{ml_data.get('approval_probability', 0)*100:.1f}%")
            st.metric("ML Recommended Loan", f"GHS {ml_data.get('recommended_loan', 0):,.0f}")
    
    # ============================================================
    # Combined Recommendation
    # ============================================================
    st.markdown("---")
    st.subheader("🎯 Combined Recommendation")
    
    final_data = result.get("final", {})
    final_decision = final_data.get("decision", "N/A")
    reasoning = final_data.get("reasoning", "")
    
    if final_decision == "APPROVE":
        st.success(f"## ✅ FINAL DECISION: {final_decision}")
        st.balloons()
    elif final_decision == "DECLINE":
        st.error(f"## ❌ FINAL DECISION: {final_decision}")
    else:
        st.warning(f"## ⚠️ FINAL DECISION: {final_decision}")
    
    st.info(f"**Reasoning:** {reasoning}")
    
    # ============================================================
    # Risk Factors
    # ============================================================
    risk_factors = final_data.get("risk_factors", [])
    if risk_factors:
        st.subheader("📌 Key Risk Factors")
        
        positive_factors = [f for f in risk_factors if f.get("impact") == "positive"]
        negative_factors = [f for f in risk_factors if f.get("impact") == "negative"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if negative_factors:
                st.markdown("**🔴 Negative Factors:**")
                for f in negative_factors:
                    st.error(f"• {f.get('factor')}")
            else:
                st.success("No significant negative factors detected")
        
        with col2:
            if positive_factors:
                st.markdown("**🟢 Positive Factors:**")
                for f in positive_factors:
                    st.success(f"• {f.get('factor')}")
    
    # ============================================================
    # Summary Card
    # ============================================================
    st.markdown("---")
    st.subheader("📋 Assessment Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; text-align:center">
            <strong>DTI Assessment</strong><br>
            <span style="font-size:24px; font-weight:bold">{dti_data.get('decision', 'N/A')}</span><br>
            <small>GHS {dti_data.get('available_monthly', 0):,.0f} available</small>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col2:
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; text-align:center">
            <strong>ML Assessment</strong><br>
            <span style="font-size:24px; font-weight:bold">{ml_data.get('decision', 'N/A')}</span><br>
            <small>{ml_data.get('default_probability', 0)*100:.0f}% default risk</small>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col3:
        color = "green" if final_decision == "APPROVE" else "red" if final_decision == "DECLINE" else "orange"
        st.markdown(f"""
        <div style="background-color:{color}20; padding:15px; border-radius:10px; text-align:center">
            <strong>Final Decision</strong><br>
            <span style="font-size:24px; font-weight:bold; color:{color}">{final_decision}</span><br>
            <small>{'Proceed with approval' if final_decision == 'APPROVE' else 'Do not approve' if final_decision == 'DECLINE' else 'Manual review required'}</small>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()