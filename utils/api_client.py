"""
API Client for FastAPI Backend
Connects Streamlit dashboard to the backend API
"""

import requests
import streamlit as st

# Backend API URL
API_BASE_URL = "http://localhost:8000"


class APIClient:
    """Client for communicating with FastAPI backend"""
    
    def __init__(self, base_url=API_BASE_URL):
        self.base_url = base_url
    
    def _request(self, method, endpoint, data=None):
        """Make HTTP request to backend"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=10)
            else:
                return None
            
            if response.status_code == 200:
                result = response.json()
                return result.get("data")
            else:
                st.error(f"API Error: {response.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Make sure API is running (python run.py)")
            return None
        except Exception as e:
            st.error(f"Request failed: {e}")
            return None
    
    # ============================================================
    # Analytics Endpoints
    # ============================================================
    
    def get_dashboard_summary(self):
        """Get dashboard summary statistics"""
        return self._request("GET", "/analytics/summary")
    
    def get_npl_trend(self, months=12):
        """Get NPL trend over time"""
        return self._request("GET", f"/analytics/npl-trend?months={months}")
    
    def get_lender_performance(self):
        """Get lender performance rankings"""
        return self._request("GET", "/analytics/lender-performance")
    
    def get_loan_purpose_analysis(self):
        """Get default rates by loan purpose"""
        return self._request("GET", "/analytics/loan-purpose")
    
    # ============================================================
    # Borrower Endpoints
    # ============================================================
    
    def get_borrowers(self, limit=100):
        """Get list of all borrowers"""
        return self._request("GET", f"/borrowers?limit={limit}")
    
    def get_borrower(self, borrower_id):
        """Get detailed borrower information"""
        return self._request("GET", f"/borrowers/{borrower_id}")
    
    def get_borrower_loans(self, borrower_id):
        """Get all loans for a borrower"""
        return self._request("GET", f"/borrowers/{borrower_id}/loans")
    
    # ============================================================
    # Prediction Endpoints
    # ============================================================
    
    def predict_default(self, borrower_id, loan_amount=None):
        """Predict default probability for a borrower"""
        data = {"borrower_id": borrower_id}
        if loan_amount:
            data["loan_amount"] = loan_amount
        return self._request("POST", "/predict", data)
    
    # ============================================================
    # Loan Stacking Endpoints
    # ============================================================
    
    def check_stacking(self, borrower_id):
        """Check for same-day loan stacking"""
        data = {"borrower_id": borrower_id}
        return self._request("POST", "/stacking/check", data)
    
    # ============================================================
    # Affordability Endpoints
    # ============================================================
    
    def calculate_affordability(self, monthly_income, total_debt):
        """Calculate affordable loan amount"""
        data = {"monthly_income": monthly_income, "total_debt": total_debt}
        return self._request("POST", "/affordability/calculate", data)
    
    def get_borrower_affordability(self, borrower_id):
        """Get affordability for a specific borrower"""
        return self._request("GET", f"/affordability/borrower/{borrower_id}")


# Singleton instance
api_client = APIClient()


def check_backend_health():
    """Check if backend API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False