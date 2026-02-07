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
    
    try:
        # Initialize data loader and service
        loader = DataLoader(data_dir)
        service = MenuEngineeringService(loader)
        
        logger.info("\nLOADING DATA...")
        # Load and prepare data
        order_items = service.load_menu_data(menu_items_file, order_items_file, place_id)
        
        logger.info("\nBUILDING INSIGHTS...")
        # Perform menu engineering analysis
        insights = service.build_menu_insights(order_items)
        
        # Generate summary
        summary = service.build_summary(insights)
        
        logger.info("\nGENERATING PRICING RECOMMENDATIONS...")
        # Analyze pricing
        pricing_recs = service.analyze_pricing_optimization(order_items)
        
        logger.info("\nGENERATING OPTIMIZATION PLAN...")
        # Generate optimization plan
        plan = service.generate_optimization_plan(insights, pricing_recs)
        
        # Export insights
        logger.info(f"\nEXPORTING RESULTS TO: {output_file}")
        service.export_insights(insights, output_file)
        
        # Print summary report
        print("\n" + "="*70)
        print("MENU ENGINEERING ANALYSIS - SUMMARY REPORT")
        print("="*70)
        print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nTotal Items Analyzed: {summary['total_items']}")
        print(f"Total Revenue: DKK {summary['total_revenue']:,.2f}")
        print(f"Total Cost: DKK {summary['total_cost']:,.2f}")
        print(f"Total Margin: DKK {summary['total_margin']:,.2f}")
        print(f"Average Margin %: {summary['average_margin_pct']:.1f}%")
        
        print("\n" + "-"*70)
        print("ITEMS BY CATEGORY")
        print("-"*70)
        
        for category in ["star", "plowhorse", "puzzle", "dog"]:
            category_label = {
                "star": "Star STARS",
                "plowhorse": "Plowhorse PLOWS",
                "puzzle": "Puzzle PUZZLES",
                "dog": " DOGS",
            }.get(category, category.upper())
            
            if category in summary["by_category"]:
                cat_summary = summary["by_category"][category]
                print(f"\n{category_label}:")
                print(f"  Items: {cat_summary['count']}")
                print(f"  Revenue: DKK {cat_summary['revenue']:,.2f}")
                print(f"  Margin: DKK {cat_summary['margin']:,.2f}")
        
        print("\n" + "-"*70)
        print("OPTIMIZATION PLAN")
        print("-"*70)
        print(f"\nExpected Revenue Uplift: DKK {plan.expected_revenue_uplift:,.2f}")
        print(f"Expected Margin Improvement: {plan.expected_margin_uplift:.1f}%")
        print(f"\nItems to Promote: {len(plan.items_to_promote)}")
        print(f"Items to Reposition: {len(plan.items_to_optimize)}")
        print(f"Items to Reprice: {len(plan.items_to_reprice)}")
        print(f"Items to Remove: {len(plan.items_to_remove)}")
        
        print("\n" + "-"*70)
        print("IMPLEMENTATION PRIORITY")
        print("-"*70)
        for i, action in enumerate(plan.implementation_priority, 1):
            print(f"{i}. {action}")
        
        print("\n" + "="*70)
        print(f"Yes Analysis complete. Insights exported to: {output_file}")
        print("="*70 + "\n")
        
        logger.info("CLI analysis completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        print("\n[Warning]  Data files not found!")
        print("Please ensure the following files are in the 'data/' directory:")
        print("  - Menu Engineering Part 2/dim_menu_items.csv")
        print("  - Menu Engineering Part 2/fct_order_items.csv")
        print("\nDownload datasets from: https://github.com/ynakhla/DIH-X-AUC-Hackathon/releases")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\n[Error] Error during analysis: {e}")
        sys.exit(1)


def run_dashboard() -> None:
    """
    Launch the Streamlit dashboard for interactive analysis.
    
    The dashboard provides a web-based interface for:
    - Viewing menu performance analytics
    - Exploring profitability and popularity metrics
    - Analyzing pricing opportunities
    - Generating and downloading optimization plans
    """
    import subprocess
    
    logger.info("Launching Streamlit dashboard...")
    
    # Run Streamlit dashboard
    try:
        subprocess.run(
            ["streamlit", "run", "src/streamlit_dashboard.py"],
            check=True
        )
    except FileNotFoundError:
        logger.error("Streamlit not found. Install with: pip install streamlit")
        print("\n[Error] Streamlit not installed.")
        print("Install with: pip install streamlit")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Streamlit error: {e}")
        sys.exit(1)


def main() -> None:
    """
    Main entry point with CLI argument parsing.
    
    Provides two modes:
    1. CLI Mode: Batch analysis with CSV output
    2. Dashboard Mode (default): Interactive web interface
    
    Usage:
        python src/main.py --cli                 # CLI mode
        python src/main.py                       # Dashboard mode (default)
        streamlit run src/streamlit_dashboard.py # Direct dashboard launch
    """
    parser = argparse.ArgumentParser(
        description="MenuMetrics Intelligence Platform - Data-driven menu optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CLI analysis with defaults
  python src/main.py --cli
  
  # Run CLI analysis and save to custom file
  python src/main.py --cli --output my_results.csv
  
  # Launch interactive dashboard (default)
  python src/main.py
  
  # Direct Streamlit launch
  streamlit run src/streamlit_dashboard.py
        """
    )
    
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode (batch analysis with CSV output)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="menu_insights.csv",
        help="Output CSV file path for CLI mode (default: menu_insights.csv)",
    )
    
    args = parser.parse_args()
    
    # Set output environment variable if provided
    if args.output:
        os.environ["MENU_OUTPUT_FILE"] = args.output
    
    # Route to appropriate mode
    if args.cli:
        run_cli_analysis()
    else:
        print("\n" + "="*70)
        print("  MenuMetrics Intelligence Platform - Dashboard Mode")
        print("="*70)
        print("\nLaunching interactive dashboard...")
        print("Open http://localhost:8501 in your browser")
        print("\nPress Ctrl+C to stop the server")
        print("="*70 + "\n")
        
        run_dashboard()

