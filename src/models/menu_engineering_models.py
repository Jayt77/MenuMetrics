"""
File: menu_engineering_models.py
Description: Data models for comprehensive menu engineering analysis.
Dependencies: dataclasses, typing, enum
Author: FlavorCraft Menu Intelligence Team

This module defines all data structures used in the menu engineering analysis pipeline.
Each model is designed for specific analysis phases and follows industry-standard
classifications (BCG Matrix, contribution margin analysis, price elasticity).

Key Models:
- MenuItemMetrics: Core profitability and popularity metrics
- MenuItemInsight: Business intelligence and recommendations
- PricingOptimization: Price elasticity and optimization suggestions
- CustomerBehavior: Purchase pattern analysis
- MenuOptimizationPlan: Complete menu restructuring recommendations
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


class MenuQuadrant(Enum):
    """
    BCG Matrix-style menu quadrants for strategic classification.
    
    - STAR: High profitability + High popularity (Protect & promote)
    - PLOW: High profitability + Low popularity (Promote/reposition)
    - PUZZLE: Low profitability + High popularity (Optimize pricing)
    - DOG: Low profitability + Low popularity (Eliminate/redesign)
    """
    STAR = "Star"
    PLOW = "Plow"
    PUZZLE = "Puzzle"
    DOG = "Dog"


@dataclass
class MenuItemMetrics:
    """
    Core aggregated metrics for a menu item.
    
    This model holds the fundamental profitability and popularity metrics
    computed from raw transaction data. Used as input for classification
    and optimization analysis.

    Attributes:
        menu_item_id (str): Unique identifier for the menu item
        title (str): Human-readable menu item name
        total_quantity (float): Total units sold across all transactions
        total_revenue (float): Total revenue generated in DKK
        total_cost (float): Total cost of goods sold (COGS) in DKK
        contribution_margin (float): Revenue minus COGS in DKK
        margin_percentage (float): Contribution margin as percentage of revenue
        popularity_score (float): Normalized popularity score (0-100)
        category (str): Menu engineering quadrant (Star, Plow, Puzzle, Dog)
        price_avg (float): Average selling price per unit in DKK
        price_std (float): Price standard deviation (indicates discounting)
        
    Example:
        >>> item = MenuItemMetrics(
        ...     menu_item_id="101",
        ...     title="Caesar Salad",
        ...     total_quantity=1500,
        ...     total_revenue=52500,  # 1500 * 35 DKK avg
        ...     total_cost=15750,     # ~30% COGS
        ...     contribution_margin=36750,
        ...     margin_percentage=70.0,
        ...     popularity_score=85.5,
        ...     category="Star"
        ... )
    """

    menu_item_id: str
    title: str
    total_quantity: float
    total_revenue: float
    total_cost: float
    contribution_margin: float
    margin_percentage: float
    popularity_score: float
    category: str
    price_avg: float = 0.0
    price_std: float = 0.0
    suggested_action: Optional[str] = None

