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


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a number as a percentage.
    
    Args:
        value (float): Decimal value (e.g., 0.75 for 75%)
        decimals (int): Decimal places (default: 1)
    
    Returns:
        str: Formatted percentage like "75.0%"
    
    Example:
        >>> format_percentage(0.75)
        '75.0%'
        >>> format_percentage(0.666667, decimals=2)
        '66.67%'
    """
    if not isinstance(value, (int, float)):
        return "N/A%"
    
    return f"{value*100:.{decimals}f}%"


def format_quantity(quantity: float, decimals: int = 0) -> str:
    """
    Format a quantity with thousand separators.
    
    Args:
        quantity (float): Quantity value
        decimals (int): Decimal places (default: 0)
    
    Returns:
        str: Formatted quantity like "1,234"
    
    Example:
        >>> format_quantity(1234567.89)
        '1,234,568'
        >>> format_quantity(1000.5, decimals=1)
        '1,000.5'
    """
    if not isinstance(quantity, (int, float)):
        return "N/A"
    
    return f"{quantity:,.{decimals}f}"


# ========== Statistical Calculations ==========

def calculate_contribution_margin(
    revenue: float,
    cost_of_goods: float
) -> Tuple[float, float]:
    """
    Calculate contribution margin and margin percentage.
    
    Contribution Margin = Revenue - Cost of Goods Sold (COGS)
    Margin % = (Contribution Margin / Revenue) × 100
    
    Args:
        revenue (float): Total revenue in DKK
        cost_of_goods (float): Total COGS in DKK
    
    Returns:
        Tuple[float, float]: (contribution_margin, margin_percentage)
    
    Example:
        >>> margin, pct = calculate_contribution_margin(1000, 300)
        >>> margin
        700.0
        >>> pct
        70.0
    """
    margin = revenue - cost_of_goods
    
    if revenue == 0:
        pct = 0.0
    else:
        pct = (margin / revenue) * 100
    
    return margin, pct


def calculate_roi(
    profit: float,
    investment: float
) -> float:
    """
    Calculate return on investment (ROI).
    
    ROI = (Profit / Investment) × 100
    
    Args:
        profit (float): Profit in DKK
        investment (float): Investment amount in DKK
    
    Returns:
        float: ROI percentage
    
    Example:
        >>> calculate_roi(500, 1000)
        50.0
    """
    if investment == 0:
        return 0.0
    
    return (profit / investment) * 100


def calculate_elasticity(
    price_change_pct: float,
    quantity_change_pct: float
) -> float:
    """
    Calculate price elasticity of demand.
    
    Elasticity = % Change in Quantity / % Change in Price
    
    Interpretation:
    - |E| < 1.0: Inelastic (demand insensitive to price)
    - |E| = 1.0: Unit elastic
    - |E| > 1.0: Elastic (demand sensitive to price)
    
    Args:
        price_change_pct (float): Percentage change in price
        quantity_change_pct (float): Percentage change in quantity
    
    Returns:
        float: Elasticity coefficient
    
    Example:
        >>> calculate_elasticity(10, -8)  # 10% price increase -> 8% quantity decrease
        -0.8  # Inelastic
    """
    if price_change_pct == 0:
        return 0.0
    
    return quantity_change_pct / price_change_pct


# ========== Aggregation Functions ==========

def aggregate_by_category(
    df: pd.DataFrame,
    category_col: str,
    metrics: Dict[str, str]
) -> pd.DataFrame:
    """
    Aggregate metrics by category.
    
    Args:
        df (pd.DataFrame): DataFrame with metrics
        category_col (str): Column to group by
        metrics (Dict[str, str]): Map of {output_col: (input_col, agg_func)}
    
    Returns:
        pd.DataFrame: Aggregated results by category
    
    Example:
        >>> df_agg = aggregate_by_category(df, 'category', {
        ...     'total_revenue': ('revenue', 'sum'),
        ...     'item_count': ('item_id', 'count'),
        ... })
    """
    return df.groupby(category_col).agg(**{
        k: (v[0], v[1]) for k, v in metrics.items()
    }).reset_index()


def calculate_running_total(
    df: pd.DataFrame,
    value_col: str,
    sort_col: str = None,
    ascending: bool = False
) -> pd.Series:
    """
    Calculate cumulative sum (running total) of a column.
    
    Useful for ABC analysis (Pareto principle) where you want to find
    which items account for 80% of sales, etc.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        value_col (str): Column to sum
        sort_col (str): Column to sort by (default: value_col)
        ascending (bool): Sort direction
    
    Returns:
        pd.Series: Cumulative sum values
    
    Example:
        >>> df_sorted = df.sort_values('revenue', ascending=False)
        >>> cumsum = calculate_running_total(df_sorted, 'revenue')
        >>> cumsum_pct = (cumsum / cumsum.iloc[-1]) * 100
    """
    if sort_col is None:
        sort_col = value_col
    
    sorted_df = df.sort_values(sort_col, ascending=ascending)
    return sorted_df[value_col].cumsum()


def identify_outliers(
    series: pd.Series,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.Series:
    """
    Identify outliers in a series.
    
    Args:
        series (pd.Series): Data series
        method (str): 'iqr' (interquartile range) or 'zscore'
        threshold (float): For IQR: typically 1.5. For zscore: typically 3.0
    
    Returns:
        pd.Series: Boolean series (True = outlier)
    
    Example:
        >>> outliers = identify_outliers(df['price'])
        >>> df[~outliers]  # Remove outliers
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < Q1 - threshold * IQR) | (series > Q3 + threshold * IQR)
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(series.dropna()))
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ========== Report Generation ==========

def generate_summary_report(
    insights_df: pd.DataFrame,
    summary_dict: Dict[str, Any]
) -> str:
    """
    Generate a text-based summary report.
    
    Args:
        insights_df (pd.DataFrame): Menu insights DataFrame
        summary_dict (Dict): Summary statistics
    
    Returns:
        str: Formatted report text
    
    Example:
        >>> report = generate_summary_report(insights, summary)
        >>> print(report)
    """
    report = """

         MENU ENGINEERING ANALYSIS - SUMMARY REPORT                 


PERFORMANCE METRICS
"""
    
    report += f"\nTotal Items:           {summary_dict['total_items']:.0f}"
    report += f"\nTotal Revenue:         {format_dkk(summary_dict['total_revenue'])}"
    report += f"\nTotal Cost:            {format_dkk(summary_dict['total_cost'])}"
    report += f"\nTotal Margin:          {format_dkk(summary_dict['total_margin'])}"
    report += f"\nAverage Margin %:      {format_percentage(summary_dict['average_margin_pct']/100)}"
    
    report += "\n\nITEMS BY CATEGORY\n" + "-" * 50
    
    for category in ['star', 'plowhorse', 'puzzle', 'dog']:
        if category in summary_dict['by_category']:
            cat_data = summary_dict['by_category'][category]
            report += f"\n{category.upper()}: {cat_data['count']} items"
            report += f"\n  Revenue: {format_dkk(cat_data['revenue'])}"
            report += f"\n  Margin:  {format_dkk(cat_data['margin'])}"
    
    return report


# ========== Data Validation ==========

def validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    raise_error: bool = True
) -> bool:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): List of required column names
        raise_error (bool): If True, raise KeyError on missing columns
    
    Returns:
        bool: True if all columns present, False otherwise
    
    Raises:
        KeyError: If raise_error=True and columns missing
    
    Example:
        >>> validate_dataframe_columns(df, ['id', 'revenue'])
        True
    """
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        msg = f"Missing columns: {missing}. Available: {list(df.columns)}"
        if raise_error:
            raise KeyError(msg)
        return False
    
    return True



def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculates the percentage change between two values.
    
    Args:
        old_value (float): The original value.
        new_value (float): The new value.
    
    Returns:
        float: Percentage change (positive for increase, negative for decrease).
    
    Raises:
        ValueError: If old_value is zero.
    """
    if old_value == 0:
        raise ValueError("Cannot calculate percentage change when old value is zero")
    
    return ((new_value - old_value) / old_value) * 100


def filter_by_date_range(df: pd.DataFrame, date_column: str, 
                         start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filters a DataFrame by a date range.
    
    Args:
        df (pd.DataFrame): The DataFrame to filter.
        date_column (str): Name of the date column.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    return df[mask]


def categorize_performance(value: float, thresholds: dict) -> str:
    """
    Categorizes a performance metric based on defined thresholds.
    
    Args:
        value (float): The value to categorize.
        thresholds (dict): Dictionary with 'low', 'medium', 'high' threshold values.
    
    Returns:
        str: Performance category ('poor', 'fair', 'good', 'excellent').
    
    Example:
        >>> categorize_performance(75, {'low': 50, 'medium': 70, 'high': 90})
        'good'
    """
    if value < thresholds['low']:
        return 'poor'
    elif value < thresholds['medium']:
        return 'fair'
    elif value < thresholds['high']:
        return 'good'
    else:
        return 'excellent'


def aggregate_by_period(df: pd.DataFrame, date_column: str, 
                       value_column: str, period: str = 'D') -> pd.DataFrame:
    """
    Aggregates data by time period (daily, weekly, monthly).
    
    Args:
        df (pd.DataFrame): The DataFrame to aggregate.
        date_column (str): Name of the date column.
        value_column (str): Name of the value column to aggregate.
        period (str): Period for aggregation ('D' for daily, 'W' for weekly, 'M' for monthly).
    
    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    
    aggregated = df[value_column].resample(period).sum().reset_index()
    
    return aggregated


def format_currency(amount: float, currency: str = 'DKK') -> str:
    """
    Formats a monetary value with currency symbol.
    
    All monetary values in the dataset are in DKK (Danish Krone).
    
    Args:
        amount (float): The monetary amount.
        currency (str): Currency code (default: 'DKK').
    
    Returns:
        str: Formatted currency string.
    
    Example:
        >>> format_currency(1250.50)
        'DKK 1,250.50'
    """
    return f"{currency} {amount:,.2f}"
