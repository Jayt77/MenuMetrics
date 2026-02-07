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


def run_cli_analysis() -> None:
    """
    Run menu engineering analysis in CLI mode.
    
    This mode loads data, performs analysis, and outputs results to CSV.
    All configuration can be provided via environment variables or defaults.
    
    Workflow:
    1. Load menu items and order data
    2. Aggregate metrics by item
    3. Classify into BCG Matrix quadrants
    4. Generate recommendations
    5. Export to CSV
    6. Print summary to console
    
    Environment Variables:
        DATA_DIR: Root data directory (default: "data")
        MENU_ITEMS_FILE: Path to dim_menu_items CSV
        MENU_ORDER_ITEMS_FILE: Path to fct_order_items CSV
        MENU_PLACE_ID: Optional place_id to filter
        MENU_OUTPUT_FILE: Optional output CSV path
    
    Raises:
        FileNotFoundError: If data files not found
        ValueError: If data format is invalid
    """
    logger.info("="*70)
    logger.info("FLAVORCRAFT MENU INTELLIGENCE - CLI MODE")
    logger.info("="*70)
    
    # Load configuration from environment or defaults
    data_dir = os.getenv("DATA_DIR", "data")
    menu_items_file = os.getenv(
        "MENU_ITEMS_FILE",
        "Menu Engineering Part 2/dim_menu_items.csv",
    )
    order_items_file = os.getenv(
        "MENU_ORDER_ITEMS_FILE",
        "Menu Engineering Part 2/fct_order_items.csv",
    )
    place_id = os.getenv("MENU_PLACE_ID")
    output_file = os.getenv("MENU_OUTPUT_FILE", "menu_insights.csv")
    
    logger.info(f"Data Directory: {data_dir}")
    logger.info(f"Menu Items: {menu_items_file}")
    logger.info(f"Order Items: {order_items_file}")
    if place_id:
        logger.info(f"Filter by Place ID: {place_id}")
    logger.info(f"Output File: {output_file}")
    logger.info("="*70)
    
