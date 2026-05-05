import streamlit as st
import pandas as pd
import plotly.express as px
from utils.queries import SUMMARY_STATS_QUERY, NPL_TREND_QUERY

def show(db):
    st.header("Executive Summary")
    st.markdown("---")
    
    stats_result = db.execute_query(SUMMARY_STATS_QUERY)
    if stats_result:
        stats = stats_result[0]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Loans", f"{stats['total_loans']:,}")
        with col2:
            default_rate = (stats['total_defaults'] / stats['total_loans'] * 100) if stats['total_loans'] > 0 else 0
            st.metric("Default Rate", f"{default_rate:.1f}%")
        with col3:
            st.metric("Avg Loan Amount", f"GHS {stats['avg_loan_amount']:,}")
        with col4:
            recent_rate = (stats['defaults_last_30d'] / stats['loans_last_30d'] * 100) if stats['loans_last_30d'] > 0 else 0
            st.metric("30-Day Default Rate", f"{recent_rate:.1f}%")
    
    st.subheader("NPL Trend")
    npl_data = db.execute_query(NPL_TREND_QUERY, (12,))
    if npl_data:
        df = pd.DataFrame(npl_data)
        fig = px.line(df, x='month', y='npl_ratio', markers=True, title="NPL Ratio Trend (Last 12 Months)")
        fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Target (15%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No NPL trend data available")
