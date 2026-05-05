import streamlit as st
import pandas as pd
from utils.queries import LENDER_PERFORMANCE_QUERY, LOAN_PURPOSE_QUERY

def show(db):
    st.header("System Alerts & Recommendations")
    st.markdown("---")
    
    lender_data = db.execute_query(LENDER_PERFORMANCE_QUERY)
    purpose_data = db.execute_query(LOAN_PURPOSE_QUERY)
    
    alerts_found = False
    
    if lender_data:
        df = pd.DataFrame(lender_data)
        high_risk = df[df['default_rate'] > 20]
        if not high_risk.empty:
            alerts_found = True
            st.error(f"CRITICAL: {len(high_risk)} lender(s) have default rates above 20%")
            for _, lender in high_risk.iterrows():
                st.write(f"- {lender['lender_name']}: {lender['default_rate']:.1f}%")
    
    if purpose_data:
        df = pd.DataFrame(purpose_data)
        high_risk = df[df['default_rate_pct'] > 25]
        if not high_risk.empty:
            alerts_found = True
            st.warning(f"WARNING: {len(high_risk)} loan purpose(s) have default rates above 25%")
            for _, purpose in high_risk.iterrows():
                st.write(f"- {purpose['loan_purpose']}: {purpose['default_rate_pct']:.1f}%")
    
    if not alerts_found:
        st.success("No active alerts. System is operating within normal parameters.")
    
    st.markdown("---")
    st.subheader("Recommended Actions")
    st.markdown("""
    1. Monitor NPL ratio daily and report to management
    2. Schedule quarterly risk reviews with high-default lenders
    3. Tighten underwriting criteria for high-risk loan purposes
    4. Continue using real-time loan stacking prevention
    """)
