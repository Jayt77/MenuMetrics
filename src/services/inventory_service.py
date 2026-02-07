"""
File: inventory_service.py
Description: Business logic for inventory management and demand forecasting.
Dependencies: pandas, numpy, sklearn
Author: Sample Team

This is a sample file demonstrating proper code structure and documentation.
Students should replace this with their actual implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class InventoryService:
    """
    Handles inventory management operations and demand forecasting.
    
    This service provides methods for analyzing inventory levels, predicting demand,
    and generating recommendations for stock optimization.
    
    Attributes:
        inventory_data (pd.DataFrame): Current inventory dataset.
        sales_data (pd.DataFrame): Historical sales dataset.
    
    Methods:
        predict_demand(item_id, period): Predicts demand for a specific item.
        calculate_reorder_point(item_id): Calculates optimal reorder point.
        identify_expiring_items(days_threshold): Identifies items near expiration.
    """
    
    def __init__(self, inventory_data: pd.DataFrame, sales_data: pd.DataFrame):
        """
        Initialize the InventoryService.
        
        Args:
            inventory_data (pd.DataFrame): Current inventory dataset.
            sales_data (pd.DataFrame): Historical sales dataset.
        """
        self.inventory_data = inventory_data
        self.sales_data = sales_data
    
    def predict_demand(self, item_id: str, period: str = 'daily') -> float:
        """
        Predicts demand for a specific item based on historical data.
        
        This is a simple moving average implementation. Students should implement
        more sophisticated forecasting methods (ARIMA, Prophet, ML models).
        
        Args:
            item_id (str): The unique identifier of the item.
            period (str): Time period for prediction ('daily', 'weekly', 'monthly').
        
        Returns:
            float: Predicted demand quantity.
        
        Raises:
            ValueError: If item_id not found in sales data.
        """
        # Filter sales data for the specific item
        item_sales = self.sales_data[self.sales_data['item_id'] == item_id]
        
        if item_sales.empty:
            raise ValueError(f"No sales data found for item: {item_id}")
        
        # Simple moving average (last 7 days for daily, etc.)
        window = 7 if period == 'daily' else 4 if period == 'weekly' else 3
        avg_demand = item_sales['quantity'].tail(window).mean()
        
        return round(avg_demand, 2)
    
