"""
Universal Data Adapter for MenuMetrics Intelligence Platform
Automatically detects, maps, and validates uploaded restaurant data.

Author: MenuMetrics Intelligence Platform Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    """Profile information for a single column."""
    name: str
    dtype: str
    cardinality: int
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: List[Any]
    is_numeric: bool
    is_datetime: bool
    is_categorical: bool
    suggested_type: str


@dataclass
class DatasetProfile:
    """Complete profile of an uploaded dataset."""
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    detected_type: str  # 'transaction', 'master', 'event', 'financial'
    quality_score: float
    completeness: float
    has_timestamps: bool
    has_ids: bool
    memory_mb: float


@dataclass
class ColumnMapping:
    """Mapping between uploaded column and standard schema."""
    uploaded_name: str
    standard_name: str
    confidence: float
    transformation: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    quality_score: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class UniversalDataAdapter:
    """
    Intelligent data adapter that automatically processes uploaded restaurant data.

    Capabilities:
    - Schema detection and profiling
    - Intelligent column mapping
    - Data quality validation
    - Type inference and conversion
    - Adaptive processing based on data structure
    """

    # Standard schemas for different data types
    STANDARD_SCHEMAS = {
        'transaction_data': {
            'required': ['menu_item_id', 'quantity'],
            'optional': ['timestamp', 'unit_price', 'unit_cost', 'customer_id', 'order_id'],
            'column_patterns': {
                'menu_item_id': ['item_id', 'product_id', 'dish_id', 'menu_id', 'sku'],
                'menu_item_title': ['item_name', 'product_name', 'dish_name', 'title', 'name', 'item'],
                'quantity': ['qty', 'sold_qty', 'units', 'amount', 'sold'],
                'unit_price': ['price', 'selling_price', 'amount', 'unit_price', 'sale_price'],
                'unit_cost': ['cost', 'cogs', 'ingredient_cost', 'food_cost'],
                'timestamp': ['date', 'created', 'order_date', 'datetime', 'time', 'created_at'],
                'customer_id': ['user_id', 'customer', 'client_id'],
                'order_id': ['transaction_id', 'receipt', 'order_number']
            }
        },
        'menu_master': {
            'required': ['item_id', 'title'],
            'optional': ['category', 'price', 'cost', 'description', 'status'],
            'column_patterns': {
                'item_id': ['id', 'product_id', 'menu_item_id', 'sku'],
                'title': ['name', 'item_name', 'product_name', 'dish_name'],
                'category': ['section', 'type', 'category_name', 'group'],
                'price': ['selling_price', 'unit_price', 'retail_price'],
                'cost': ['cogs', 'ingredient_cost', 'unit_cost'],
                'description': ['desc', 'details', 'info'],
                'status': ['active', 'enabled', 'availability']
            }
        }
    }

    def __init__(self):
        """Initialize the universal data adapter."""
        self.profiles: Dict[str, DatasetProfile] = {}
        self.mappings: Dict[str, List[ColumnMapping]] = {}
        logger.info("UniversalDataAdapter initialized")

    def profile_dataset(self, df: pd.DataFrame, dataset_name: str = "uploaded_data") -> DatasetProfile:
        """
        Analyze uploaded data and create comprehensive profile.

        Args:
            df: Uploaded DataFrame
            dataset_name: Identifier for the dataset

        Returns:
            DatasetProfile with complete metadata
        """
        logger.info(f"Profiling dataset: {dataset_name} ({len(df):,} rows, {len(df.columns)} columns)")

        # Profile each column
        column_profiles = {}
        for col in df.columns:
            column_profiles[col] = self._profile_column(df[col])

        # Detect dataset type
        detected_type = self._detect_dataset_type(df, column_profiles)

        # Calculate quality metrics
        quality_score = self._calculate_quality_score(df, column_profiles)
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100

        # Check for special column types
        has_timestamps = any(cp.is_datetime for cp in column_profiles.values())
        has_ids = any('id' in cp.name.lower() for cp in column_profiles.values())

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        profile = DatasetProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns=column_profiles,
            detected_type=detected_type,
            quality_score=quality_score,
            completeness=completeness,
            has_timestamps=has_timestamps,
            has_ids=has_ids,
            memory_mb=memory_mb
        )

        self.profiles[dataset_name] = profile
        logger.info(f"Profile complete: type={detected_type}, quality={quality_score:.1f}/100")

        return profile

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        """Profile a single column."""
        null_count = series.isnull().sum()
        null_percentage = (null_count / len(series)) * 100
        unique_count = series.nunique()
        cardinality = unique_count / len(series) if len(series) > 0 else 0

        # Type detection
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        is_categorical = cardinality < 0.5 and unique_count < 100

        # Suggested type
        if is_datetime:
            suggested_type = 'datetime'
        elif is_numeric:
            if cardinality > 0.9:
                suggested_type = 'numeric_id'
            elif unique_count < 20:
                suggested_type = 'categorical'
            else:
                suggested_type = 'numeric'
        elif is_categorical:
            suggested_type = 'categorical'
        else:
            suggested_type = 'text'

        # Sample values (non-null)
        sample_values = series.dropna().head(5).tolist()

        return ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            cardinality=cardinality,
            null_count=int(null_count),
            null_percentage=null_percentage,
            unique_count=unique_count,
            sample_values=sample_values,
            is_numeric=is_numeric,
            is_datetime=is_datetime,
            is_categorical=is_categorical,
            suggested_type=suggested_type
        )

    def _detect_dataset_type(self, df: pd.DataFrame, profiles: Dict[str, ColumnProfile]) -> str:
        """
        Detect whether data is transaction, master, event, or financial.

        Heuristics:
        - Transaction: High cardinality, timestamps, numeric amounts, IDs
        - Master: Low-medium cardinality, descriptive fields
        - Event: Timestamps, user actions, event names
        - Financial: Payment fields, amounts, currencies
        """
        has_timestamps = any(p.is_datetime for p in profiles.values())
        has_amounts = any('price' in p.name.lower() or 'amount' in p.name.lower()
                         for p in profiles.values())
        has_quantity = any('qty' in p.name.lower() or 'quantity' in p.name.lower()
                          for p in profiles.values())
        has_events = any('event' in p.name.lower() for p in profiles.values())
        has_payment = any('payment' in p.name.lower() or 'card' in p.name.lower()
                         for p in profiles.values())

        avg_cardinality = np.mean([p.cardinality for p in profiles.values()])

        if has_payment:
            return 'financial'
        elif has_events and has_timestamps:
            return 'event'
        elif has_timestamps and has_amounts and has_quantity:
            return 'transaction'
        elif avg_cardinality < 0.3:
            return 'master'
        else:
            return 'transaction'  # Default assumption

    def _calculate_quality_score(self, df: pd.DataFrame, profiles: Dict[str, ColumnProfile]) -> float:
        """
        Calculate overall data quality score (0-100).

        Factors:
        - Completeness (no nulls)
        - Consistency (appropriate types)
        - Validity (reasonable values)
        """
        # Completeness score
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100

        # Type consistency score
        type_scores = []
        for profile in profiles.values():
            if profile.suggested_type == 'numeric' and profile.is_numeric:
                type_scores.append(100)
            elif profile.suggested_type == 'datetime' and profile.is_datetime:
                type_scores.append(100)
            elif profile.suggested_type in ['categorical', 'text']:
                type_scores.append(80)
            else:
                type_scores.append(50)

        type_consistency = np.mean(type_scores) if type_scores else 50

        # Combined score
        quality_score = (completeness * 0.6 + type_consistency * 0.4)

        return min(100, max(0, quality_score))

