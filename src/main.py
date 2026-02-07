"""
File: main.py
Description: Entry point for MenuMetrics Intelligence Platform - supports CLI and dashboard modes.
Dependencies: os, sys, argparse, pandas, logging
Author: MenuMetrics Intelligence Platform Team

This module provides the primary entry point for menu engineering analysis.

Usage (CLI mode):
    python src/main.py --cli [--output results.csv]

Usage (Dashboard mode):
    streamlit run src/streamlit_dashboard.py

Environment Variables:
    DATA_DIR: Root data directory (default: "data")
    MENU_ITEMS_FILE: Path to dim_menu_items CSV
    MENU_ORDER_ITEMS_FILE: Path to fct_order_items CSV
    MENU_PLACE_ID: Optional place_id to filter
    MENU_OUTPUT_FILE: Optional output CSV path
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from src.models.data_loader import DataLoader
from src.services.menu_engineering_service import MenuEngineeringService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
