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
    
