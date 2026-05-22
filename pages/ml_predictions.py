"""
ML Predictions Page - Uses Trained Stacked Ensemble Model
"""

import streamlit as st
import requests
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"

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

def show(db=None):
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
        borrower_dict = {b["full_name"]: b["borrower_id"] for b in borrowers}
        selected = st.selectbox("Select Borrower", list(borrower_dict.keys()))
        
        if st.button("Predict Default Risk", type="primary", use_container_width=True):
            with st.spinner("Calling trained stacked ensemble model..."):
                result = api_post("/predict", {"borrower_id": borrower_dict[selected]})
            
            if result:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Default Probability", 
                        f"{result.get('default_probability', 0)*100:.1f}%"
                    )
                    st.metric("Decision", result.get("decision", "N/A"))
                
                with col2:
                    st.metric("Risk Level", result.get("risk_level", "N/A"))
                    st.metric(
                        "Recommended Loan", 
                        f"GHS {result.get('recommended_loan_amount', 0):,.0f}"
                    )
                
                # Gauge chart
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
                st.error("Prediction failed. Make sure backend is running.")
    else:
        st.warning("No borrowers found. Please generate sample data.")

if __name__ == "__main__":
    show()