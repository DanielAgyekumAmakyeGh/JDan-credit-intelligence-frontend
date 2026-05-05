import streamlit as st
from datetime import datetime
from utils.queries import LOAN_STACKING_QUERY

def show(db):
    st.header("Loan Stacking Prevention")
    st.markdown("---")
    
    st.warning("""
    **What is Loan Stacking?**
    Loan stacking occurs when a borrower takes multiple loans from different lenders on the same day.
    This dashboard checks for same-day loans to prevent over-indebtedness.
    """)
    
    st.subheader("Check Borrower")
    
    borrower_input = st.text_input("Enter Borrower ID", placeholder="e.g., 12345")
    
    if st.button("Check for Stacking", type="primary"):
        if borrower_input:
            loans_today = db.execute_query(LOAN_STACKING_QUERY, (borrower_input,))
            count = loans_today[0]['loans_today'] if loans_today else 0
            
            if count > 0:
                st.error(f"ALERT: Borrower has {count} other loan(s) today!")
                st.warning("Recommendation: Delay approval and investigate")
            else:
                st.success(f"No same-day loans found for borrower {borrower_input}")
                st.info("Recommendation: Proceed with approval")
        else:
            st.warning("Please enter a Borrower ID")
    
    st.caption(f"Check performed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
