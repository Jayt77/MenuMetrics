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

