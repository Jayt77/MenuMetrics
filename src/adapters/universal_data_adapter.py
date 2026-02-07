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
