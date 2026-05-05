import streamlit as st
import pandas as pd
import plotly.express as px
from utils.queries import LENDER_PERFORMANCE_QUERY

def show(db):
    st.header("Lender Performance")
    st.markdown("---")
    
    lender_data = db.execute_query(LENDER_PERFORMANCE_QUERY)
    
    if lender_data:
        df = pd.DataFrame(lender_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Default Rate", f"{df['default_rate'].mean():.1f}%")
        with col2:
            worst = df.loc[df['default_rate'].idxmax(), 'lender_name'] if len(df) > 0 else "N/A"
            st.metric("Highest Default", f"{df['default_rate'].max():.1f}%", delta=worst)
        with col3:
            best = df.loc[df['default_rate'].idxmin(), 'lender_name'] if len(df) > 0 else "N/A"
            st.metric("Lowest Default", f"{df['default_rate'].min():.1f}%", delta=best)
        
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
        
        high_risk = df[df['default_rate'] > 20]
        if not high_risk.empty:
            st.warning(f"Warning: {len(high_risk)} lender(s) have default rates above 20%")
    else:
        st.info("No lender performance data available")
