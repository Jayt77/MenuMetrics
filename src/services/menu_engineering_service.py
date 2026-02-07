"""
File: menu_engineering_service.py
Description: Comprehensive menu engineering analytics and optimization service.
Dependencies: pandas, numpy, scipy, scikit-learn
Author: MenuMetrics Intelligence Platform Team

This service implements the complete menu engineering pipeline:
1. Data aggregation and profitability analysis
2. BCG Matrix classification (Stars, Plows, Puzzles, Dogs)
3. Pricing optimization using elasticity analysis
4. Customer behavior and bundling opportunity detection
5. Menu optimization plan generation with actionable recommendations

The service transforms raw sales data into strategic business intelligence
for menu optimization, pricing decisions, and promotional strategies.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from models.data_loader import DataLoader
from models.menu_engineering_models import (
    MenuItemMetrics,
    MenuItemInsight,
    PricingOptimization,
    CustomerBehavior,
    MenuOptimizationPlan,
    MenuQuadrant,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MenuEngineeringService:
    """
    Enterprise menu engineering analytics service.
    
    This service provides:
    - Profitability and popularity analysis
    - BCG Matrix (4-quadrant) classification
    - Price elasticity and optimization recommendations
    - Customer behavior and bundling analysis
    - Complete menu optimization plan generation
    
    All analysis follows industry-standard restaurant management practices
    and uses contribution margin and sales volume as primary metrics.
    
    Attributes:
        data_loader (DataLoader): Data loading utility
        currency (str): Currency code (default: DKK)
        profit_threshold (float): Median contribution margin threshold
        popularity_threshold (float): Median quantity threshold
    """

    def __init__(self, data_loader: DataLoader, currency: str = "DKK") -> None:
        """
        Initialize the menu engineering service.
        
        Args:
            data_loader (DataLoader): Utility for loading CSV data
            currency (str): Currency label for reporting. Defaults to "DKK".
        """
        self.data_loader = data_loader
        self.currency = currency
        self.profit_threshold: float = 0.0
        self.popularity_threshold: float = 0.0
        self.insights_df: Optional[pd.DataFrame] = None
        self.menu_items_df: Optional[pd.DataFrame] = None  # Store for satisfaction metrics
        logger.info(f"MenuEngineeringService initialized (Currency: {currency})")

    def load_menu_data(
        self,
        menu_items_file: str,
        order_items_file: str,
        place_id: Optional[str] = None,
        profitability_margin: float = 0.20,
    ) -> pd.DataFrame:
        """
        Load menu and order data from CSV files and prepare unified dataset.

        This method:
        1. Loads dim_menu_items and fct_order_items CSVs
        2. Normalizes column names and data types
        3. Merges datasets on menu_item_id
        4. Filters by place_id if specified

        Args:
            menu_items_file (str): Path to dim_menu_items CSV
            order_items_file (str): Path to fct_order_items CSV
            place_id (Optional[str]): Filter for specific location/merchant
            profitability_margin (float): Assumed profit margin for items without cost data (default: 0.20)

        Returns:
            pd.DataFrame: Merged and normalized order data with menu metadata

        Raises:
            FileNotFoundError: If CSV files not found

        Example:
            >>> service = MenuEngineeringService(loader)
            >>> orders = service.load_menu_data(
            ...     "Menu Engineering Part 2/dim_menu_items.csv",
            ...     "Menu Engineering Part 2/fct_order_items.csv"
            ... )
        """
        try:
            menu_items = self.data_loader.load_csv(menu_items_file)
            order_items = self.data_loader.load_csv(order_items_file)
            return self.prepare_order_items(menu_items, order_items, place_id, profitability_margin)
        except FileNotFoundError as e:
            logger.error(f"Failed to load menu data: {e}")
            raise

    def prepare_order_items(
        self,
        menu_items: pd.DataFrame,
        order_items: pd.DataFrame,
        place_id: Optional[str] = None,
        profitability_margin: float = 0.20,
    ) -> pd.DataFrame:
        """
        Normalize and prepare order items for analysis.

        This method handles flexible column naming conventions found in real-world
        datasets. It:
        - Normalizes column names (accepts multiple naming variants)
        - Fills missing cost data with configurable profitability assumption
        - Merges order and menu data
        - Calculates derived metrics (revenue, cost, contribution)
        - Filters by place_id if provided

        Supported column name variations:
            - Item ID: menu_item_id, item_id, product_id, menu_item
            - Quantity: quantity, qty, item_quantity
            - Price: price, unit_price, item_price
            - Cost: cost, unit_cost, item_cost
            - Place: place_id, location_id, merchant_id

        Args:
            menu_items (pd.DataFrame): Menu items master data
            order_items (pd.DataFrame): Order transaction data
            place_id (Optional[str]): Filter for specific location
            profitability_margin (float): Assumed profit margin for items without cost data (default: 0.20 = 20%)

        Returns:
            pd.DataFrame: Normalized and merged order data

        Raises:
            ValueError: If required columns cannot be resolved
        """
        logger.info("Preparing and normalizing order items...")

        # Store menu_items for later satisfaction metrics loading
        self.menu_items_df = menu_items

        # Resolve flexible column naming for menu items
        menu_id_col = self._resolve_column(menu_items, ["id", "menu_item_id", "item_id"])
        menu_title_col = self._resolve_column(menu_items, ["title", "name", "menu_item_title"])
        menu_status_col = self._resolve_column(
            menu_items,
            ["status", "is_active", "available"],
            optional=True,
        )

        # Resolve flexible column naming for order items
        order_menu_id_col = self._resolve_column(
            order_items,
            ["menu_item_id", "item_id", "product_id", "menu_item"],
        )
        order_qty_col = self._resolve_column(order_items, ["quantity", "qty", "item_quantity"])
        order_price_col = self._resolve_column(order_items, ["price", "unit_price", "item_price"])
        order_cost_col = self._resolve_column(
            order_items,
            ["cost", "unit_cost", "item_cost"],
            optional=True,
        )
        order_place_col = self._resolve_column(
            order_items,
            ["place_id", "location_id", "merchant_id"],
            optional=True,
        )
        order_title_col = self._resolve_column(
            order_items,
            ["title", "item_name", "menu_item_title"],
            optional=True,
        )

        # Standardize order items column names
        normalized = order_items.copy()
        normalized = normalized.rename(
            columns={
                order_menu_id_col: "menu_item_id",
                order_qty_col: "quantity",
                order_price_col: "unit_price",
            }
        )

        # Handle optional columns
        if order_cost_col:
            # Some datasets store line-total cost in a generic "cost"/"item_cost" column.
            # Preserve raw values first and normalize to unit cost after quantity conversion.
            if order_cost_col in ["cost", "item_cost"]:
                normalized = normalized.rename(columns={order_cost_col: "cost_raw"})
            else:
                normalized = normalized.rename(columns={order_cost_col: "unit_cost"})
        else:
            # Industry standard: estimate COGS at 30% of selling price
            normalized["unit_cost"] = 0.0

        if order_place_col:
            normalized = normalized.rename(columns={order_place_col: "place_id"})

        if order_title_col:
            normalized = normalized.rename(columns={order_title_col: "item_title"})
        else:
            normalized["item_title"] = np.nan

        # Filter by place_id if provided
        if place_id and "place_id" in normalized.columns:
            original_count = len(normalized)
            normalized = normalized[normalized["place_id"].astype(str) == str(place_id)]
            logger.info(f"Filtered to place {place_id}: {len(normalized):,} of {original_count:,} rows")

        # Prepare menu subset for merge
        menu_subset = menu_items[[menu_id_col, menu_title_col]].copy()
        menu_subset = menu_subset.rename(
            columns={menu_id_col: "menu_item_id", menu_title_col: "menu_title"}
        )

        if menu_status_col:
            menu_subset["menu_status"] = menu_items[menu_status_col]
        else:
            menu_subset["menu_status"] = "unknown"

        # Merge order and menu data
        merged = normalized.merge(menu_subset, on="menu_item_id", how="left")
        merged["item_title"] = merged["item_title"].fillna(merged["menu_title"])

        # Type conversion and validation
        merged["unit_price"] = pd.to_numeric(merged["unit_price"], errors='coerce')
        merged["quantity"] = pd.to_numeric(merged["quantity"], errors='coerce')
        if "cost_raw" in merged.columns:
            merged["cost_raw"] = pd.to_numeric(merged["cost_raw"], errors='coerce')
            merged["unit_cost"] = np.where(
                merged["quantity"] > 0,
                merged["cost_raw"] / merged["quantity"],
                np.nan,
            )
        else:
            merged["unit_cost"] = pd.to_numeric(merged["unit_cost"], errors='coerce')

        # If provided cost mirrors sales price too closely, it is likely not COGS.
        # Fall back to estimated COGS from profitability assumptions in that case.
        valid_cost = merged[
            (merged["unit_price"] > 0)
            & (merged["quantity"] > 0)
            & merged["unit_cost"].notna()
        ]
        if len(valid_cost) >= 1000:
            price_ratio = valid_cost["unit_cost"] / valid_cost["unit_price"]
            near_parity = ((price_ratio >= 0.95) & (price_ratio <= 1.05)).mean()
            median_ratio = float(price_ratio.median())
            if near_parity >= 0.90 and 0.95 <= median_ratio <= 1.05:
                logger.warning(
                    "Cost column appears to mirror sales prices (likely non-COGS). "
                    "Ignoring provided cost and estimating COGS from profitability assumptions."
                )
                merged["unit_cost"] = np.nan

        # Handle missing costs with configurable profitability assumption
        # If profitability_margin is 20%, cost = 80% of price
        cost_multiplier = 1.0 - profitability_margin
        if merged["unit_cost"].isna().sum() > 0:
            logger.warning(
                f"Estimated cost for {merged['unit_cost'].isna().sum():,} items "
                f"(assumed {profitability_margin*100:.0f}% profit margin, cost = {cost_multiplier*100:.0f}% of price)"
            )
            merged["unit_cost"] = merged["unit_cost"].fillna(merged["unit_price"] * cost_multiplier)
        
        # Calculate derived metrics
        merged["item_revenue"] = merged["unit_price"] * merged["quantity"]
        merged["item_cost"] = merged["unit_cost"] * merged["quantity"]

        # DATA QUALITY FILTERING: Remove admin, test, and invalid items
        initial_count = len(merged)

        # Filter 1: Remove inactive items
        if "menu_status" in merged.columns:
            merged = merged[
                (merged["menu_status"].str.lower() == "active") |
                (merged["menu_status"] == "unknown")
            ]
            logger.info(f"Filtered out {initial_count - len(merged):,} inactive items")

        # Filter 2: Remove items with blank/null titles
        merged = merged[
            merged["item_title"].notna() &
            (merged["item_title"].str.strip() != "")
        ]
        logger.info(f"Filtered out items with blank titles (remaining: {len(merged):,})")

        # Filter 3: Remove items with invalid prices (≤0)
        merged = merged[merged["unit_price"] > 0]
        logger.info(f"Filtered out items with price ≤ 0 (remaining: {len(merged):,})")

        # Filter 4: Remove test/admin/unspecified items by title keywords
        test_keywords = ['test', 'admin', 'sample', 'demo', 'do not', 'donotuse',
                        'placeholder', 'temp', 'draft', 'delete', 'xxx', 'zzz',
                        'unspecified', 'unknown', 'n/a', 'null', 'none', 'default',
                        'blank', 'empty', 'tbd', 'to be determined']
        test_pattern = '|'.join(test_keywords)
        merged = merged[~merged["item_title"].str.lower().str.contains(test_pattern, na=False)]
        logger.info(f"Filtered out test/admin/unspecified items (remaining: {len(merged):,})")

        # Filter 5: Remove items with 0 quantity (data quality issue)
        merged = merged[merged["quantity"] > 0]
        logger.info(f"Filtered out items with 0 quantity (remaining: {len(merged):,})")

        total_filtered = initial_count - len(merged)
        if total_filtered > 0:
            logger.info(f"✓ Data quality filtering complete: Removed {total_filtered:,} ({total_filtered/initial_count*100:.1f}%) invalid/test items")

        logger.info(f"Prepared {len(merged):,} clean order items for analysis")
        return merged

