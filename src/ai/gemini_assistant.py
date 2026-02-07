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

    def _create_data_context(self):
        """Create a summary of the data for AI context"""
        if self.classification_df is None or len(self.classification_df) == 0:
            self.data_context = "No classification data available."
            return

        df = self.classification_df

        # Overall stats
        total_items = len(df)
        total_restaurants = df['place_id'].nunique() if 'place_id' in df.columns else 0
        total_revenue = df['revenue'].sum() if 'revenue' in df.columns else 0
        total_orders = df['order_count'].sum() if 'order_count' in df.columns else 0

        # Category breakdown
        if 'category' in df.columns:
            category_stats = df.groupby('category').agg({
                'item_id': 'count',
                'revenue': 'sum',
                'order_count': 'sum'
            })
        else:
            category_stats = pd.DataFrame()

        # Build list of available datasets
        dataset_info = "\n".join([f"   - {name}: {len(data):,} rows, columns: {', '.join(data.columns.tolist()[:10])}"
                                  for name, data in list(self.datasets.items())[:15]])

        # Build context string
        self.data_context = f"""
You are a menu engineering expert analyzing restaurant data. Here's the dataset overview:

CLASSIFICATION SUMMARY:
- Total menu items: {total_items:,}
- Total restaurants: {total_restaurants}
- Total revenue: {total_revenue:,.0f} DKK
- Total orders: {total_orders:,}

MENU ENGINEERING CATEGORIES:
The items are classified into 4 categories based on profitability and popularity:

1. STARS (High Profit + High Popularity): Best performers - promote heavily, maintain quality

2. PLOWHORSES (Low Profit + High Popularity): Popular but less profitable - consider price increases

3. PUZZLES (High Profit + Low Popularity): Profitable but underordered - need better marketing

4. DOGS (Low Profit + Low Popularity): Remove or redesign these items

CUSTOMER SATISFACTION METRICS:
- Customer ratings are available on a 0-5 star scale
- Items have associated vote counts indicating reliability
- High-rated items (4+ stars) should be prioritized in recommendations
- Low-rated items (<3 stars) may have quality issues even if profitable

AVAILABLE DATASETS ({len(self.datasets)} total):
{dataset_info}

CRITICAL RULES:
1. **NEVER make up SQL queries** - You don't have database access, only the context data provided above
2. **NEVER invent products, items, or data** - Only reference items explicitly shown in the context
3. **ONLY use data provided in the context** - If you see timeline data with specific items and counts, USE THAT EXACT DATA
4. **If you don't have specific information, say "I don't have that data" instead of guessing**
5. **Timeline data format**: When you see "12:00 [Pizza (78), Pasta (54)]", those are REAL items with REAL counts from the data
6. **NEVER fabricate**: Don't make up items not in the provided context
7. **RESTAURANT-SPECIFIC DATA ONLY**: When answering about a specific restaurant, ONLY reference items from that restaurant's context

NEVER say you don't have access to these datasets. You DO have them. Use them to answer questions!

When answering questions:
- Provide specific numbers and examples from the data
- Reference multiple datasets when relevant
- Calculate insights using the available data
- Give actionable recommendations
- Use the menu engineering framework to explain insights
"""
