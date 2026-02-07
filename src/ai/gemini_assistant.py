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

    def __init__(self, api_key: str, classification_file: str = '../menu_classification_results.csv'):
        """
        Initialize the Groq assistant with data

        Args:
            api_key: Groq API key
            classification_file: Path to classification results CSV
        """
        logger.info("Initializing Groq Menu Assistant...")

        if not api_key:
            raise ValueError("GROQ_API_KEY is required")

        # Configure Groq
        self.client = Groq(api_key=api_key)
        self.model_name = 'llama-3.1-8b-instant'  # 8K context

        # Load ALL available datasets
        self.datasets = {}

        # Load classification results (main dataset)
        try:
            self.classification_df = pd.read_csv(classification_file)
            self.datasets['classification'] = self.classification_df
            logger.info(f"   Loaded classification: {len(self.classification_df):,} items")
        except Exception as e:
            logger.warning(f"   Could not load classification file: {e}")
            self.classification_df = None

        # Load order timeline data (timestamps + restaurants)
        try:
            self.timeline_df = pd.read_csv('../order_timeline_data.csv')
            self.datasets['order_timeline'] = self.timeline_df
            logger.info(f"   Loaded order timeline: {len(self.timeline_df):,} orders with timestamps")
        except Exception as e:
            logger.info(f"   Timeline data not available: {e}")
            self.timeline_df = None

        # Load all CSV files from BOTH Menu Engineering directories
        data_dirs = ['../data/Menu Engineering Part 1', '../data/Menu Engineering Part 2']
        payments_parts = []

        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                try:
                    for filename in os.listdir(data_dir):
                        if filename.endswith('.csv'):
                            name = filename.replace('.csv', '')
                            filepath = os.path.join(data_dir, filename)
                            try:
                                # Only load first 100k rows for large files to save memory
                                file_size = os.path.getsize(filepath)
                                if file_size > 50_000_000:  # > 50MB
                                    df = pd.read_csv(filepath, nrows=100000)
                                    logger.info(f"   Loaded {name}: {len(df):,} rows (sampled from large file)")
                                else:
                                    df = pd.read_csv(filepath)
                                    logger.info(f"   Loaded {name}: {len(df):,} rows")

                                # Special handling for payment parts - merge them
                                if 'fct_payments_part' in name:
                                    payments_parts.append(df)
                                else:
                                    self.datasets[name] = df

                            except Exception as e:
                                logger.warning(f"   Skipped {filename}: {e}")
                except Exception as e:
                    logger.warning(f"   Could not load from {data_dir}: {e}")

        # Merge payment parts into single dataset
        if payments_parts:
            self.datasets['fct_payments'] = pd.concat(payments_parts, ignore_index=True)
            logger.info(f"   Merged payments: {len(self.datasets['fct_payments']):,} total rows")

        # Create data context
        self._create_data_context()

        logger.info(f"Groq Assistant ready! {len(self.datasets)} datasets loaded")

