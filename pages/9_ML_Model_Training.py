"""
ML Model Training Dashboard
Train and evaluate stacked ensemble model
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.db_connection import get_db
from utils.stacked_model import stacked_model
import time

st.set_page_config(page_title="ML Model Training", page_icon="🤖", layout="wide")

def format_percentage(value):
    try:
        return f"{float(value):.1f}%"
    except:
        return "0%"

def show(db):
    st.header("🤖 Stacked ML Model Training")
    st.markdown("---")
    
    st.info("""
    **Stacked Ensemble Model Architecture:**
    