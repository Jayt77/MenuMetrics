#!/bin/bash
echo "Starting MenuMetrics Dashboard..."
cd src
streamlit run streamlit_dashboard.py --theme.base=light
