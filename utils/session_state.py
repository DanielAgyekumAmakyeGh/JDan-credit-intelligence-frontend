"""
Session state manager for global variables across dashboard pages
"""

import streamlit as st

class SessionState:
    """Manage global session variables"""
    
    @staticmethod
    def init():
        """Initialize all session state variables"""
        if 'borrower_id' not in st.session_state:
            st.session_state.borrower_id = None
        if 'borrower_name' not in st.session_state:
            st.session_state.borrower_name = None
        if 'borrower_data' not in st.session_state:
            st.session_state.borrower_data = None
        if 'selected_page' not in st.session_state:
            st.session_state.selected_page = "Executive Summary"
        if 'backend_connected' not in st.session_state:
            st.session_state.backend_connected = False
    
    @staticmethod
    def set_borrower(borrower_id, borrower_name, borrower_data=None):
        """Set the current selected borrower globally"""
        st.session_state.borrower_id = borrower_id
        st.session_state.borrower_name = borrower_name
        st.session_state.borrower_data = borrower_data
    
    @staticmethod
    def get_borrower_id():
        """Get current borrower ID"""
        return st.session_state.borrower_id
    
    @staticmethod
    def get_borrower_name():
        """Get current borrower name"""
        return st.session_state.borrower_name
    
    @staticmethod
    def get_borrower_data():
        """Get current borrower data"""
        return st.session_state.borrower_data
    
    @staticmethod
    def clear_borrower():
        """Clear the selected borrower"""
        st.session_state.borrower_id = None
        st.session_state.borrower_name = None
        st.session_state.borrower_data = None
    
    @staticmethod
    def set_page(page_name):
        """Set the current page"""
        st.session_state.selected_page = page_name
    
    @staticmethod
    def get_page():
        """Get the current page"""
        return st.session_state.selected_page

# Initialize on import
SessionState.init()