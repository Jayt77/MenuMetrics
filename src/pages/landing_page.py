"""
Landing Page for MenuMetrics Intelligence Platform
Welcome screen with data upload and quick-start guide.

Author: MenuMetrics Intelligence Platform Team
"""

import streamlit as st
import pandas as pd
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.universal_data_adapter import UniversalDataAdapter


def show_landing_page() -> bool:
    """
    Display the landing page with data upload.

    Returns:
        True if data was successfully uploaded and processed
    """
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px 40px 20px;'>
        <h1 style='font-size: 4em; font-weight: 700; color: #1f2937; margin-bottom: 20px;'>
            Welcome to MenuMetrics
        </h1>
        <p style='font-size: 1.8em; color: #6b7280; font-weight: 300; margin-bottom: 40px;'>
            Get menu optimizations in minutes
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Value Proposition
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3 style='color: #4f46e5; margin-bottom: 15px;'>Data-Driven Insights</h3>
            <p>BCG Matrix analysis, profitability metrics, and performance indicators</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3 style='color: #4f46e5; margin-bottom: 15px;'>Instant Results</h3>
            <p>Upload your data and get actionable recommendations in seconds</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3 style='color: #4f46e5; margin-bottom: 15px;'>Smart Automation</h3>
            <p>Automatically detects and maps your data structure</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Data Upload Section
    st.markdown("""
    <div style='text-align: center; padding: 30px 20px 20px 20px;'>
        <h2 style='font-size: 2.5em; color: #1f2937;'>Get Started</h2>
        <p style='font-size: 1.2em; color: #6b7280; margin-bottom: 30px;'>
            Upload your restaurant data to begin analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Upload Options
    upload_col1, upload_col2 = st.columns([2, 1])

    with upload_col1:
        st.subheader("Option 1: Upload Your Own Data")

        st.info("Upload one or more CSV files containing your menu and transaction data")

        uploaded_files = st.file_uploader(
            "Upload CSV Files (you can select multiple files)",
            type=['csv'],
            accept_multiple_files=True,
            help="Select all CSV files you want to analyze. You can upload order data, menu items, payments, etc.",
            key="files_upload"
        )

        if uploaded_files and len(uploaded_files) > 0:
            st.write(f"**{len(uploaded_files)} file(s) uploaded:**")
            for file in uploaded_files:
                st.write(f"- {file.name}")

            if st.button("Process Uploaded Data", type="primary", use_container_width=True):
                success = process_uploaded_files(uploaded_files)
                if success:
                    import time
                    st.success("Data processed successfully! Redirecting to dashboard...")
                    st.balloons()
                    time.sleep(1.0)  # Show balloons animation
                    st.rerun()

    with upload_col2:
        st.subheader("Option 2: Use Sample Data")

        st.write("Try MenuMetrics with our sample restaurant dataset")

        if st.button("Load Sample Data", use_container_width=True):
            success = load_sample_data()
            if success:
                import time
                st.success("Sample data loaded! Redirecting to dashboard...")
                time.sleep(0.5)  # Brief pause to show success message
                st.rerun()

