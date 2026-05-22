import streamlit as st
import pandas as pd
import plotly.express as px
from utils.api_client import api_client

def show(db=None):  # db parameter kept for compatibility
    st.header("Executive Summary")
    st.markdown("---")
    
    # Get data from API
    summary = api_client.get_dashboard_summary()
    
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Loans", summary.get('total_loans', 0))
        with col2:
            st.metric("Default Rate", f"{summary.get('default_rate', 0)}%")
        with col3:
            st.metric("Avg Loan Amount", f"GHS {summary.get('avg_loan_amount', 0):,.0f}")
        with col4:
            st.metric("Active Borrowers", summary.get('active_borrowers', 0))
    else:
        st.warning("Could not load dashboard summary. Make sure backend is running.")
    
    # NPL Trend
    st.subheader("NPL Ratio Trend")
    npl_data = api_client.get_npl_trend(12)
    
    if npl_data:
        df = pd.DataFrame(npl_data)
        fig = px.line(df, x='month', y='npl_ratio', markers=True, title="Monthly NPL Trend")
        fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Target (15%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No NPL trend data available")