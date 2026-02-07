"""
Landing Page for MenuMetrics Intelligence Platform
Welcome screen with data upload and quick-start guide.

Author: MenuMetrics Intelligence Platform Team
"""

import streamlit as st
import pandas as pd
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.universal_data_adapter import UniversalDataAdapter


