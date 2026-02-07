"""
Groq AI-Powered Menu Engineering Assistant
Adapted for MenuMetrics Streamlit Dashboard

Uses Groq's API for natural language understanding of menu data
"""
import pandas as pd
from groq import Groq
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


class GeminiMenuAssistant:
    """AI assistant powered by Groq for menu insights"""
