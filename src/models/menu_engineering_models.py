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


@dataclass
class MenuItemInsight:
    """
    Business intelligence and strategic recommendations for a menu item.
    
    This model extends MenuItemMetrics with contextual analysis and
    actionable recommendations based on profitability and popularity.
    Used for stakeholder communication and decision support.

    Attributes:
        menu_item_id (str): Unique item identifier
        title (str): Item name
        quadrant (MenuQuadrant): Strategic classification
        metrics (MenuItemMetrics): Core metrics
        profit_rank (int): Ranking by contribution margin (1 = highest)
        popularity_rank (int): Ranking by sales volume (1 = highest)
        trend (str): Direction of metric change ("up", "down", "stable")
        recommendation (str): Strategic recommendation text
        revenue_opportunity (float): Potential revenue uplift in DKK
        action_priority (str): Priority level ("high", "medium", "low")
        
    Example:
        >>> insight = MenuItemInsight(
        ...     menu_item_id="101",
        ...     title="Caesar Salad",
        ...     quadrant=MenuQuadrant.STAR,
        ...     metrics=menu_metrics,
        ...     profit_rank=3,
        ...     popularity_rank=2,
        ...     trend="up",
        ...     recommendation="Maintain positioning. Consider premium variant.",
        ...     revenue_opportunity=5000,
        ...     action_priority="medium"
        ... )
    """
    menu_item_id: str
    title: str
    quadrant: MenuQuadrant
    metrics: MenuItemMetrics
    profit_rank: int
    popularity_rank: int
    trend: str = "stable"
    recommendation: str = ""
    revenue_opportunity: float = 0.0
    action_priority: str = "medium"


@dataclass
class PricingOptimization:
    """
    Price elasticity and optimization analysis for a menu item.
    
    Analyzes price sensitivity based on historical price variation
    and quantity sold. Recommends optimal pricing to maximize revenue
    or profit based on business objectives.

    Attributes:
        menu_item_id (str): Item identifier
        title (str): Item name
        current_price (float): Current average price in DKK
        optimal_price (float): Recommended price in DKK
        price_elasticity (float): Estimated elasticity coefficient
        price_change_percent (float): Suggested change as percentage
        revenue_impact (float): Estimated revenue change in DKK
        profit_impact (float): Estimated profit change in DKK
        confidence_level (float): Confidence in recommendation (0-1)
        rationale (str): Explanation of price recommendation
        
    Example:
        >>> pricing = PricingOptimization(
        ...     menu_item_id="101",
        ...     title="Caesar Salad",
        ...     current_price=35.0,
        ...     optimal_price=42.0,
        ...     price_elasticity=-0.8,  # 1% price increase → 0.8% quantity decrease
        ...     price_change_percent=20.0,
        ...     revenue_impact=4000,
        ...     profit_impact=6500,
        ...     confidence_level=0.72,
        ...     rationale="High margin, low elasticity suggests pricing power"
        ... )
    """
    menu_item_id: str
    title: str
    current_price: float
    optimal_price: float
    price_elasticity: float
    price_change_percent: float
    revenue_impact: float
    profit_impact: float
    confidence_level: float
    rationale: str = ""


@dataclass
class CustomerBehavior:
    """
    Customer purchase behavior analysis for a menu item.
    
    Analyzes how customers interact with menu items through purchase
    patterns, co-purchases, and seasonal trends. Identifies bundling
    and cross-sell opportunities.

    Attributes:
        menu_item_id (str): Item identifier
        title (str): Item name
        avg_units_per_transaction (float): Average quantity per order
        repeat_purchase_rate (float): Percentage of repeat customers (0-100)
        co_purchase_items (List[str]): Frequently bought together item IDs
        seasonal_index (float): Seasonal strength (1.0 = no seasonality)
        customer_segments (Dict): Segments likely to purchase this item
        description_sentiment (str): Sentiment indicators from menu descriptions
        
    Example:
        >>> behavior = CustomerBehavior(
        ...     menu_item_id="101",
        ...     title="Caesar Salad",
        ...     avg_units_per_transaction=1.2,
        ...     repeat_purchase_rate=65.0,
        ...     co_purchase_items=["102", "105"],  # Dressing, croutons
        ...     seasonal_index=1.15,  # Seasonal - higher in summer
        ...     customer_segments={"health_conscious": 0.85, "lunch_crowd": 0.72},
        ...     description_sentiment="fresh, healthy, premium"
        ... )
    """
    menu_item_id: str
    title: str
    avg_units_per_transaction: float = 1.0
    repeat_purchase_rate: float = 0.0
    co_purchase_items: List[str] = field(default_factory=list)
    seasonal_index: float = 1.0
    customer_segments: Dict[str, float] = field(default_factory=dict)
    description_sentiment: str = ""

