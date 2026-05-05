"""
XDSData Ghana - Management Dashboard
Main entry point for the modular dashboard application
"""

import streamlit as st
from config.settings import DASHBOARD_CONFIG
from utils.db_connection import get_db
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=DASHBOARD_CONFIG['title'],
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">XDSData Ghana</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{DASHBOARD_CONFIG["title"]}</div>', unsafe_allow_html=True)
    
    # Initialize database connection
    db = get_db()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### XDSData Analytics")
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Select Dashboard View",
            [
                "Executive Summary",
                "NPL Trends",
                "Lender Performance",
                "Loan Purpose Analysis",
                "Loan Stacking Check",
                "Affordability Tool",
                "Alerts & Recommendations",
                "🤖 ML Model Training",
                "🎯 ML Predictions"
            ]
        )
        
        st.markdown("---")
        st.caption(f"Last updated: {st.session_state.get('last_refresh', 'Not yet')}")
        
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Route to appropriate page
    if page == "Executive Summary":
        from pages.executive_summary import show
        show(db)
    elif page == "NPL Trends":
        from pages.npl_trends import show
        show(db)
    elif page == "Lender Performance":
        from pages.lender_performance import show
        show(db)
    elif page == "Loan Purpose Analysis":
        from pages.loan_purpose import show
        show(db)
    elif page == "Loan Stacking Check":
        from pages.loan_stacking import show
        show(db)
    elif page == "Affordability Tool":
        from pages.affordability_tool import show
        show(db)
    elif page == "Alerts & Recommendations":
        from pages.alerts import show
        show(db)
    elif page == "🤖 ML Model Training":
        from pages.ml_model_training import show
        show(db)
    elif page == "🎯 ML Predictions":
        from pages.affordability_tool import show
        show(db)
    
    # Update last refresh time
    st.session_state['last_refresh'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    main()