"""
File: data_loader.py
Description: ETL module for loading and preprocessing menu engineering data from CSV files.
Dependencies: pandas, numpy, logging
Author: MenuMetrics Intelligence Platform Team

This module follows industry-standard data loading practices with proper error handling,
data validation, and type inference. It handles the complete Menu Engineering dataset
(Part 1 & Part 2) and normalizes data for downstream analysis.

Core Responsibilities:
- Load CSV files from the data directory
- Validate data integrity and column presence
- Handle missing values and data type inference
- Merge related tables for unified analysis
- Cache loaded datasets for performance
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List

# Configure logging for data loading operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Enterprise-grade data loader for menu engineering analysis.
    
    This class implements ETL (Extract, Transform, Load) best practices:
    - Lazy loading with caching
    - Type inference and validation
    - Error handling with informative messages
    - Data quality checks
    - Memory-efficient processing
    
    Attributes:
        data_path (str): Root path to the data directory
        _cache (Dict): In-memory cache of loaded datasets
    
    Methods:
        load_csv: Load single CSV file with validation
        load_menu_items: Load and normalize menu item master data
        load_order_items: Load and normalize order transaction data
        load_payments: Load payment information data
        load_bill_of_materials: Load recipe/ingredient cost data
        load_all_menu_engineering_data: Load complete dataset stack
        merge_datasets: Merge datasets with validation
    """
    
    def __init__(self, data_path: str = "data"):
        """
        Initialize the DataLoader with a data directory path.
        
        Args:
            data_path (str): Root path to data directory. Defaults to "data".
        
        Raises:
            ValueError: If data_path does not exist.
        """
        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")
        
        self.data_path = data_path
        self._cache: Dict[str, pd.DataFrame] = {}
        logger.info(f"DataLoader initialized with path: {data_path}")
    
    def load_csv(
        self,
        filename: str,
        use_cache: bool = True,
        parse_dates: Optional[List[str]] = None,
        dtype_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Load a CSV file with caching, type inference, and validation.
        
        This method implements industry-standard practices:
        - In-memory caching to avoid redundant file I/O
        - Type inference for numeric and categorical columns
        - Missing value detection and reporting
        - File existence validation
        
        Args:
            filename (str): CSV filename or relative path from data_path
            use_cache (bool): Whether to use cached data if available. Defaults to True.
            parse_dates (List[str], optional): Columns to parse as datetime
            dtype_map (Dict[str, str], optional): Explicit dtype mapping (e.g., {'col': 'float64'})
        
        Returns:
            pd.DataFrame: Loaded dataset with inferred/specified types
        
        Raises:
            FileNotFoundError: If the file does not exist
            pd.errors.ParserError: If the CSV cannot be parsed
        
        Example:
            >>> loader = DataLoader("data")
            >>> df = loader.load_csv("Menu Engineering Part 2/dim_menu_items.csv")
            >>> print(f"Loaded {len(df)} menu items")
        """
        # Check cache first if enabled
        if use_cache and filename in self._cache:
            logger.info(f"[CACHE HIT] {filename}")
            return self._cache[filename].copy()
        
        # Construct full file path
        file_path = os.path.join(self.data_path, filename)
        
        # Validate file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Load CSV with type inference
            df = pd.read_csv(
                file_path,
                parse_dates=parse_dates,
                dtype=dtype_map,
                low_memory=False
            )
            
            # Data quality reporting
            logger.info(f"Yes Loaded {filename}: {len(df):,} rows × {len(df.columns)} columns")
            
            # Report missing values if present
            missing = df.isnull().sum()
            if missing.any():
                logger.warning(f"  Missing values: {missing[missing > 0].to_dict()}")
            
            # Cache the dataframe
            self._cache[filename] = df.copy()
            
            return df
        
        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse CSV {filename}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {filename}: {str(e)}")
            raise
    
    def load_menu_items(self) -> pd.DataFrame:
        """
        Load and normalize the menu items master data.
        
        This method standardizes the menu items data by:
        - Loading from dim_menu_items
        - Normalizing column names to lowercase with underscores
        - Converting monetary values to float
        - Handling missing values
        - Adding derived fields (is_active status)
        
        Returns:
            pd.DataFrame: Normalized menu items with columns:
                - menu_item_id: Unique item identifier
                - title: Item display name
                - price: Listed price in DKK
                - status: Active/Inactive status
                - category: Menu section/category
                
        Raises:
            FileNotFoundError: If dim_menu_items.csv not found
        
        Example:
            >>> menu_items = loader.load_menu_items()
            >>> print(f"Found {len(menu_items)} menu items")
        """
        try:
            df = self.load_csv("Menu Engineering Part 2/dim_menu_items.csv")
        except FileNotFoundError:
            raise FileNotFoundError(
                "dim_menu_items.csv not found. "
                "Please extract Menu Engineering Part 2 into data/ directory."
            )
        
        # Normalize column names to lowercase with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Ensure required columns exist
        required_cols = ['id', 'title']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dim_menu_items: {missing}")
        
        # Rename id to menu_item_id for consistency
        if 'id' in df.columns and 'menu_item_id' not in df.columns:
            df = df.rename(columns={'id': 'menu_item_id'})
        
        # Convert price to float if present
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Add active status flag
        if 'status' in df.columns:
            df['is_active'] = df['status'].str.lower() == 'active'
        else:
            df['is_active'] = True  # Default to active if no status field
        
        logger.info(f"Yes Normalized {len(df)} menu items")
        return df
    
    def load_order_items(self) -> pd.DataFrame:
        """
        Load and normalize order items transaction data.
        
        This method standardizes order transactions by:
        - Loading from fct_order_items
        - Normalizing column names
        - Converting monetary and quantity columns to numeric
        - Calculating missing cost fields if needed
        - Handling missing/invalid data
        
        Returns:
            pd.DataFrame: Normalized order items with columns:
                - order_id: Unique order identifier
                - menu_item_id: Item ordered
                - quantity: Units ordered
                - price: Sale price per unit in DKK
                - cost: Cost of goods per unit in DKK
                - total_price: quantity × price
                - total_cost: quantity × cost
        
        Raises:
            FileNotFoundError: If fct_order_items.csv not found
        
        Example:
            >>> orders = loader.load_order_items()
            >>> print(f"Loaded {len(orders):,} order line items")
        """
        try:
            df = self.load_csv("Menu Engineering Part 1/fct_order_items.csv")
        except FileNotFoundError:
            raise FileNotFoundError(
                "fct_order_items.csv not found. "
                "Please extract Menu Engineering Part 1 into data/ directory."
            )
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Map common column name variations to standard names
        column_mapping = {
            'item_id': 'menu_item_id',
            'unit_price': 'price',
            'unit_cost': 'cost',
            'qty': 'quantity',
        }
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['menu_item_id', 'quantity', 'price']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in fct_order_items: {missing}")
        
        # Convert numeric columns
        numeric_cols = ['quantity', 'price', 'cost']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate derived metrics
        df['total_price'] = df['quantity'] * df['price']
        
        if 'cost' in df.columns:
            df['total_cost'] = df['quantity'] * df['cost']
        else:
            # If cost not provided, estimate as 30% of price (industry average)
            logger.warning("'cost' column not found in order items. Using 30% of price as estimate.")
            df['cost'] = df['price'] * 0.30
            df['total_cost'] = df['quantity'] * df['cost']
        
        # Remove rows with invalid data
        df_clean = df.dropna(subset=['menu_item_id', 'quantity', 'price'])
        removed = len(df) - len(df_clean)
        if removed > 0:
            logger.warning(f"Removed {removed} rows with missing critical values")
        
        logger.info(f"Yes Normalized {len(df_clean):,} order items")
        return df_clean
    
    def load_payments(self) -> pd.DataFrame:
        """
        Load and normalize payment transaction data.
        
        This method loads payment data from both Part 1 and Part 2 files and:
        - Combines datasets if both exist
        - Normalizes column names and types
        - Converts monetary values to float
        - Handles timestamps
        
        Returns:
            pd.DataFrame: Normalized payments with columns:
                - order_id: Related order
                - amount: Payment amount in DKK
                - payment_method: Method used (cash, card, etc.)
                - timestamp: Payment time (UNIX epoch)
        
        Note:
            If both Part 1 and Part 2 files exist, they will be concatenated.
            
        Example:
            >>> payments = loader.load_payments()
            >>> print(f"Total transactions: DKK {payments['amount'].sum():,.2f}")
        """
        dfs = []
        payment_files = [
            "Menu Engineering Part 1/fct_payments_part1.csv",
            "Menu Engineering Part 2/fct_payments_part2.csv"
        ]
        
        for file in payment_files:
            try:
                df = self.load_csv(file)
                dfs.append(df)
            except FileNotFoundError:
                logger.info(f"Payment file not found: {file}")
                continue
        
        if not dfs:
            logger.warning("No payment data files found")
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Convert monetary columns
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove duplicates
        df_clean = df.drop_duplicates()
        removed = len(df) - len(df_clean)
        if removed > 0:
            logger.warning(f"Removed {removed} duplicate payment records")
        
        logger.info(f"Yes Normalized {len(df_clean):,} payment records")
        return df_clean
    
    def load_bill_of_materials(self) -> pd.DataFrame:
        """
        Load bill of materials (recipe/ingredient) data.
        
        This optional data source maps menu items to ingredients and their costs:
        - Loads from dim_bill_of_materials
        - Normalizes column names
        - Converts quantities to numeric
        
        Returns:
            pd.DataFrame: Bill of materials with columns:
                - menu_item_id: The dish
                - ingredient_id: The ingredient (item_id)
                - quantity: Amount needed
                - unit: Measurement unit
        
        Note:
            This is optional data. Returns empty DataFrame if not available.
            
        Example:
            >>> bom = loader.load_bill_of_materials()
            >>> if not bom.empty:
            ...     print(f"Recipes for {bom['menu_item_id'].nunique()} items")
        """
        try:
            df = self.load_csv("Menu Engineering Part 2/dim_bill_of_materials.csv")
        except FileNotFoundError:
            logger.info("Bill of materials data not available")
            return pd.DataFrame()
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Convert quantity column
        if 'quantity' in df.columns:
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        
        logger.info(f"Yes Loaded bill of materials for {df['menu_item_id'].nunique() if 'menu_item_id' in df.columns else 0} items")
        return df
    
    def load_all_menu_engineering_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load complete menu engineering dataset stack.
        
        This is the primary entry point for loading all menu analysis data.
        It loads and validates all available tables in a single operation.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with keys:
                - 'menu_items': Master item list
                - 'order_items': Transaction data
                - 'payments': Payment records (optional)
                - 'bill_of_materials': Recipe data (optional)
        
        Raises:
            ValueError: If critical datasets (menu_items, order_items) cannot be loaded
        
        Example:
            >>> loader = DataLoader("data")
            >>> data = loader.load_all_menu_engineering_data()
            >>> print("Loaded datasets:", data.keys())
        """
        logger.info("="*60)
        logger.info("LOADING MENU ENGINEERING DATA STACK")
        logger.info("="*60)
        
        data = {}
        
        # Load critical datasets
        try:
            data['menu_items'] = self.load_menu_items()
        except FileNotFoundError as e:
            logger.error(f"Critical: {e}")
            raise
        
        try:
            data['order_items'] = self.load_order_items()
        except FileNotFoundError as e:
            logger.error(f"Critical: {e}")
            raise
        
        # Load optional datasets
        data['payments'] = self.load_payments()
        data['bill_of_materials'] = self.load_bill_of_materials()
        
        # Summary statistics
        logger.info("="*60)
        logger.info("DATA LOAD SUMMARY")
        logger.info("="*60)
        logger.info(f"Menu Items: {len(data['menu_items']):,}")
        logger.info(f"Order Items: {len(data['order_items']):,}")
        logger.info(f"Payments: {len(data['payments']):,}")
        logger.info(f"Bill of Materials: {len(data['bill_of_materials']):,}")
        logger.info("="*60)
        
