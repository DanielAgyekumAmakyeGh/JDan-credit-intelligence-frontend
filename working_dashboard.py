import streamlit as st
import requests

st.set_page_config(page_title="Credit Intelligence Bureau", layout="wide")

API_URL = "http://127.0.0.1:8000"

with st.sidebar:
    st.title("Navigation")
    
    # Check backend
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("✅ Backend Connected")
            backend_ok = True
        else:
            st.error("❌ Backend Error")
            backend_ok = False
    except Exception as e:
        st.error(f"❌ Backend Disconnected: {e}")
        backend_ok = False
    
    page = st.radio("Select Page", ["Executive Summary", "ML Predictions"])

st.title("🏦 Credit Intelligence Bureau")

if backend_ok:
    if page == "Executive Summary":
        st.header("Executive Summary")
        
        try:
            r = requests.get(f"{API_URL}/analytics/summary")
            if r.status_code == 200:
                data = r.json().get("data", {})
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Loans", data.get("total_loans", 0))
                col2.metric("Total Borrowers", data.get("total_borrowers", 0))
                col3.metric("Total Lenders", data.get("total_lenders", 0))
                col4.metric("Default Rate", f"{data.get('default_rate', 0)}%")
            else:
                st.error(f"API Error: {r.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    elif page == "ML Predictions":
        st.header("ML Default Prediction")
        st.info("Stacked Ensemble Model - AUC: 89.4% | Recall: 96.7%")
        
        try:
            r = requests.get(f"{API_URL}/borrowers")
            if r.status_code == 200:
                borrowers = r.json().get("data", [])
                if borrowers:
                    borrower_dict = {b["full_name"]: b["borrower_id"] for b in borrowers}
                    selected = st.selectbox("Select Borrower", list(borrower_dict.keys()))
                    
                    if st.button("Predict Default Risk", type="primary"):
                        borrower_id = borrower_dict[selected]
                        pred = requests.post(f"{API_URL}/predict", json={"borrower_id": borrower_id})
                        
                        if pred.status_code == 200:
                            result = pred.json().get("data", {})
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Default Probability", f"{result.get('default_probability', 0)*100:.1f}%")
                                st.metric("Decision", result.get("decision", "N/A"))
                            with col2:
                                st.metric("Risk Level", result.get("risk_level", "N/A"))
                                st.metric("Recommended Loan", f"GHS {result.get('recommended_loan_amount', 0):,.0f}")
                            
                            factors = result.get("factors", [])
                            if factors:
                                st.subheader("Key Factors")
                                for f in factors:
                                    if f.get("impact") == "positive":
                                        st.success(f"✅ {f.get('factor')}")
                                    else:
                                        st.error(f"❌ {f.get('factor')}")
                        else:
                            st.error(f"Prediction failed: {pred.status_code}")
                else:
                    st.warning("No borrowers found")
            else:
                st.error(f"Failed to get borrowers: {r.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.warning("Backend is not running. Start it with: cd credit_intelligence_backend && python run.py")

st.markdown("---")
st.caption("Credit Intelligence Bureau - Powered by Stacked Ensemble (89.4% AUC)")