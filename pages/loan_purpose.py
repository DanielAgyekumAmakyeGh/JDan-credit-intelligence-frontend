import streamlit as st
import pandas as pd
import plotly.express as px
from utils.queries import LOAN_PURPOSE_QUERY

def show(db):
    st.header("Loan Purpose Analysis")
    st.markdown("---")
    
    purpose_data = db.execute_query(LOAN_PURPOSE_QUERY)
    
    if purpose_data:
        df = pd.DataFrame(purpose_data)
        
        fig = px.bar(df, x='loan_purpose', y='default_rate_pct',
                     title="Default Rate by Loan Purpose",
                     color='default_rate_pct',
                     color_continuous_scale='Reds',
                     text='default_rate_pct')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detailed Data")
        st.dataframe(df, use_container_width=True)
        
        high_risk = df[df['default_rate_pct'] > 25]
        if not high_risk.empty:
            st.error(f"High Risk: {high_risk['loan_purpose'].tolist()} have default rates > 25%")
    else:
        st.info("No loan purpose data available")
