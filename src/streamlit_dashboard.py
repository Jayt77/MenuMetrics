"""
File: streamlit_dashboard.py
Description: Interactive dashboard for menu engineering analysis and optimization.
Dependencies: streamlit, plotly, pandas
Author: MenuMetrics Intelligence Platform Team

This module provides a comprehensive Streamlit dashboard for stakeholders to:
1. View menu performance analytics (BCG Matrix visualization)
2. Analyze profitability and popularity metrics
3. Explore pricing optimization recommendations
4. Generate and download optimized menu plans
5. Compare current vs. optimized menu

The dashboard is the primary user interface for MenuMetrics Intelligence Platform.
Features are organized in tabs for easy navigation and stakeholder communication.

Usage:
    streamlit run src/streamlit_dashboard.py --logger.level=error
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from models.data_loader import DataLoader
from services.menu_engineering_service import MenuEngineeringService
from pages.landing_page import show_landing_page

# Configure logging (suppress verbose Streamlit logs)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="MenuMetrics Intelligence Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean Sparrow-style dashboard
st.markdown("""
<style>
    /* Force a single light theme to avoid mixed light/dark rendering */
    :root, [data-theme="dark"], [data-theme="light"] {
        color-scheme: light !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .stApp,
    html,
    body {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }

    /* Main content area - clean white background */
    .main {
        background-color: #FFFFFF;
        padding: 2rem;
    }

    /* Sidebar - light gray like Sparrow */
    [data-testid="stSidebar"] {
        background-color: #F7F8FA;
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1rem 1.5rem;
    }

    /* Sidebar navigation styling */
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        color: #1F2937;
        font-size: 14px;
        margin-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.25rem;
    }

    [data-testid="stSidebar"] .stRadio label {
        background-color: transparent;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        color: #6B7280;
        font-size: 14px;
        font-weight: 500;
        border: none;
        margin: 0;
        transition: all 0.2s;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #E5E7EB;
        color: #1F2937;
    }

    [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
    }

    /* KPI metric cards - white with border like Sparrow */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1F2937;
    }

    div[data-testid="stMetricLabel"] {
        color: #6B7280;
        font-size: 13px;
        font-weight: 500;
        text-transform: none;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 12px;
    }

    /* Status badges - cleaner style */
    .star-badge {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }

    .dog-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }

    .puzzle-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }

    .plow-badge {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }

    /* Loading overlay */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.95);
        z-index: 9999;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        backdrop-filter: blur(5px);
    }

    .loading-spinner {
        border: 4px solid #E5E7EB;
        border-top: 4px solid #1E88E5;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .loading-text {
        margin-top: 1rem;
        font-size: 16px;
        color: #374151;
        font-weight: 500;
    }

    /* Headers */
    h1, h2, h3 {
        color: #1F2937;
        font-weight: 700;
    }

    h1 {
        font-size: 32px;
        margin-bottom: 1.5rem;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 6px;
        font-weight: 500;
        transition: background-color 0.2s;
    }

    .stButton > button:hover {
        background-color: #1976D2;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ========== Session State Management ==========

def initialize_session_state():
    """Initialize Streamlit session state for data caching."""
    if "service" not in st.session_state:
        st.session_state.service = None
    if "insights" not in st.session_state:
        st.session_state.insights = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "order_items" not in st.session_state:
        st.session_state.order_items = None
    if "pricing_recs" not in st.session_state:
        st.session_state.pricing_recs = None
    if "optimization_plan" not in st.session_state:
        st.session_state.optimization_plan = None
    # Data upload session state
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    if "uploaded_menu" not in st.session_state:
        st.session_state.uploaded_menu = None
    if "all_uploaded_files" not in st.session_state:
        st.session_state.all_uploaded_files = None
    if "data_source" not in st.session_state:
        st.session_state.data_source = None  # 'uploaded', 'sample', or None
    if "data_profile" not in st.session_state:
        st.session_state.data_profile = None
    if "data_mappings" not in st.session_state:
        st.session_state.data_mappings = None
    # Pagination state
    if "show_more_counts" not in st.session_state:
        st.session_state.show_more_counts = {}
    # AI Assistant state
    if "ai_assistant" not in st.session_state:
        st.session_state.ai_assistant = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # Profitability assumptions
    if "profitability_margin" not in st.session_state:
        st.session_state.profitability_margin = 0.20  # 20% default
    # Language analysis cache
    if "language_analysis" not in st.session_state:
        st.session_state.language_analysis = None
    # Behavior analysis cache
    if "behavior_analysis" not in st.session_state:
        st.session_state.behavior_analysis = None

