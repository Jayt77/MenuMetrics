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
