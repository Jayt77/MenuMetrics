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


