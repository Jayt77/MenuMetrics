"""
File: test_helpers.py
Description: Unit tests for utility helper functions.
Dependencies: pytest, pandas
Author: Sample Team

This is a sample file demonstrating proper test structure and documentation.
Students should replace this with their actual tests.

Run tests with: pytest tests/
"""

import pytest
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.helpers import (
    convert_unix_timestamp,
    calculate_percentage_change,
    categorize_performance,
    format_currency
)


class TestHelpers:
    """Test suite for helper utility functions."""
    
    def test_convert_unix_timestamp(self):
        """Test UNIX timestamp conversion to datetime."""
        # Test known timestamp (2021-01-01 00:00:00 UTC)
        timestamp = 1609459200
        result = convert_unix_timestamp(timestamp)
        
        assert isinstance(result, datetime)
        assert result.year == 2021
        assert result.month == 1
        assert result.day == 1
    
