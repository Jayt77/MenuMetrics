"""
File: helpers.py
Description: Utility functions for menu engineering analysis and common operations.
Dependencies: pandas, datetime, typing
Author: MenuMetrics Intelligence Platform Team

This module provides utility functions for:
- Data formatting and conversion
- Timestamp handling (UNIX epoch conversion)
- Financial calculations and formatting
- Statistical analysis helpers
- Report generation utilities

All timestamp fields in the dataset are UNIX integers. Use convert_unix_timestamp()
to convert them to readable dates. All monetary values are in DKK (Danish Krone).
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union, List, Dict, Any, Optional, Tuple


# ========== Timestamp Utilities ==========

def convert_unix_timestamp(timestamp: Union[int, float]) -> datetime:
    """
    Convert UNIX timestamp to a datetime object.
    
    All timestamp fields in the Menu Engineering dataset are UNIX integers
    (seconds since epoch). Use this function to convert to readable dates.
    
    Args:
        timestamp (Union[int, float]): UNIX timestamp (seconds since epoch)
    
    Returns:
        datetime: Converted datetime object
    
    Raises:
        ValueError: If timestamp is invalid
    
    Example:
        >>> ts = 1609459200  # 2021-01-01 00:00:00 UTC
        >>> dt = convert_unix_timestamp(ts)
        >>> print(dt)
        2021-01-01 00:00:00
    """
    try:
        if not isinstance(timestamp, (int, float)):
            raise ValueError(f"Timestamp must be int or float, got {type(timestamp)}")
        return datetime.fromtimestamp(timestamp)
    except (ValueError, OSError) as e:
        raise ValueError(f"Invalid timestamp {timestamp}: {e}")


def convert_timestamp_column(
    df: pd.DataFrame,
    column_name: str,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Convert a timestamp column from UNIX to datetime.
    
    Uses pandas to_datetime for efficient vectorized conversion.
    Handles missing values gracefully.
    
    Args:
        df (pd.DataFrame): DataFrame containing timestamp column
        column_name (str): Name of column to convert
        inplace (bool): If True, modify original DataFrame. Else return copy.
    
    Returns:
        pd.DataFrame: DataFrame with converted timestamp column
    
    Raises:
        KeyError: If column does not exist
    
    Example:
        >>> df['created_at'] = 1609459200
        >>> df = convert_timestamp_column(df, 'created_at')
        >>> df['created_at'].dtype
        dtype('<M8[ns]')
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame")
    
    result = df if inplace else df.copy()
    result[column_name] = pd.to_datetime(result[column_name], unit='s', errors='coerce')
    return result


# ========== Financial Formatting ==========

def format_dkk(amount: float, decimals: int = 2) -> str:
    """
    Format a number as Danish Krone currency.
    
    All financial values in the dataset are in DKK (Danish Krone).
    
    Args:
        amount (float): Amount to format
        decimals (int): Decimal places (default: 2)
    
    Returns:
        str: Formatted string like "DKK 1,234.56"
    
    Example:
        >>> format_dkk(1234.567)
        'DKK 1,234.57'
        >>> format_dkk(1000000, decimals=0)
        'DKK 1,000,000'
    """
    if not isinstance(amount, (int, float)):
        return "DKK N/A"
    
    if decimals == 0:
        return f"DKK {amount:,.0f}"
    return f"DKK {amount:,.{decimals}f}"

