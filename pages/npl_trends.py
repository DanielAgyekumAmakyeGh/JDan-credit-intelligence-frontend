import streamlit as st
import pandas as pd
import plotly.express as px
from utils.queries import NPL_TREND_QUERY

def show(db):
    st.header("NPL Trends Analysis")
    st.markdown("---")
    
    months = st.slider("Select time range (months)", 3, 24, 12)
    
    npl_data = db.execute_query(NPL_TREND_QUERY, (months,))
    
    if npl_data:
        df = pd.DataFrame(npl_data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average NPL", f"{df['npl_ratio'].mean():.1f}%")
        with col2:
            st.metric("Peak NPL", f"{df['npl_ratio'].max():.1f}%")
        with col3:
            st.metric("Current NPL", f"{df['npl_ratio'].iloc[-1]:.1f}%")
        with col4:
            st.metric("Target", "15%", delta="Below" if df['npl_ratio'].iloc[-1] < 15 else "Above")
        
        fig = px.line(df, x='month', y='npl_ratio', markers=True, title="NPL Ratio Over Time")
        fig.add_hline(y=15, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Raw Data")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No data available for the selected period")
