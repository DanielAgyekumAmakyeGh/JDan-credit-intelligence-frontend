import streamlit as st
import pandas as pd
import plotly.express as px
from utils.api_client import api_client

def show(db=None):
    st.header("Lender Performance")
    st.markdown("---")
    
    lenders = api_client.get_lender_performance()
    
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
        
        st.subheader("Detailed Lender Data")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No lender performance data available")