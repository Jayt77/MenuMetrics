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

    def ask(self, question: str, restaurant_id: Optional[int] = None,
            classification_df: Optional[pd.DataFrame] = None) -> str:
        """
        Ask the AI assistant a question about the menu data

        Args:
            question: User's question
            restaurant_id: Optional restaurant ID to filter data
            classification_df: Optional DataFrame to use (for dashboard integration)

        Returns:
            AI assistant's answer
        """
        try:
            # Use classification data if available
            if self.classification_df is not None:
                df = self.classification_df
            else:
                return "No menu data available. Please load data first."

            # Build comprehensive statistical context
            context_prefix = "\n\n=== COMPLETE DATASET ANALYSIS ===\n"

            # Filter data if restaurant specified
            if restaurant_id and 'place_id' in df.columns:
                df = df[df['place_id'] == restaurant_id].copy()
                context_prefix += f"Restaurant ID: {restaurant_id}\n\n"

                if len(df) == 0:
                    return f"No data found for restaurant #{restaurant_id}"
            else:
                context_prefix += "ANALYZING ALL RESTAURANTS\n\n"

            # COMPREHENSIVE STATISTICS FOR ALL ITEMS
            total_items = len(df)

            context_prefix += f"FULL DATASET STATISTICS (ALL {total_items:,} items):\n"

            if 'revenue' in df.columns:
                context_prefix += f"\nREVENUE STATISTICS:\n"
                context_prefix += f"  - Total revenue: {df['revenue'].sum():,.0f} DKK\n"
                context_prefix += f"  - Mean revenue per item: {df['revenue'].mean():,.0f} DKK\n"
                context_prefix += f"  - Median revenue: {df['revenue'].median():,.0f} DKK\n"
                context_prefix += f"  - Min revenue: {df['revenue'].min():,.0f} DKK\n"
                context_prefix += f"  - Max revenue: {df['revenue'].max():,.0f} DKK\n"
                context_prefix += f"  - 25th percentile: {df['revenue'].quantile(0.25):,.0f} DKK\n"
                context_prefix += f"  - 75th percentile: {df['revenue'].quantile(0.75):,.0f} DKK\n"

            if 'order_count' in df.columns:
                context_prefix += f"\nORDER COUNT STATISTICS:\n"
                context_prefix += f"  - Total orders: {df['order_count'].sum():,.0f}\n"
                context_prefix += f"  - Mean orders per item: {df['order_count'].mean():.0f}\n"
                context_prefix += f"  - Median orders: {df['order_count'].median():.0f}\n"
                context_prefix += f"  - Min orders: {df['order_count'].min():.0f}\n"
                context_prefix += f"  - Max orders: {df['order_count'].max():.0f}\n"

            if 'price' in df.columns:
                context_prefix += f"\nPRICE STATISTICS:\n"
                context_prefix += f"  - Mean price: {df['price'].mean():.2f} DKK\n"
                context_prefix += f"  - Median price: {df['price'].median():.2f} DKK\n"
                context_prefix += f"  - Price range: {df['price'].min():.0f} - {df['price'].max():.0f} DKK\n"

            # CATEGORY-SPECIFIC STATISTICS
            if 'category' in df.columns:
                context_prefix += f"\nCATEGORY BREAKDOWN (ALL {total_items:,} items):\n"

                for category in ['Star', 'Plowhorse', 'Puzzle', 'Dog']:
                    cat_items = df[df['category'] == category]
                    if len(cat_items) > 0:
                        cat_revenue = cat_items['revenue'].sum() if 'revenue' in cat_items.columns else 0
                        cat_orders = cat_items['order_count'].sum() if 'order_count' in cat_items.columns else 0
                        cat_mean_rev = cat_items['revenue'].mean() if 'revenue' in cat_items.columns else 0
                        cat_median_rev = cat_items['revenue'].median() if 'revenue' in cat_items.columns else 0

                        context_prefix += f"\n{category.upper()}S - {len(cat_items):,} items:\n"
                        context_prefix += f"  - Total revenue: {cat_revenue:,.0f} DKK ({cat_revenue/df['revenue'].sum()*100:.1f}% of total)\n"
                        context_prefix += f"  - Total orders: {cat_orders:,.0f}\n"
                        context_prefix += f"  - Mean revenue per item: {cat_mean_rev:,.0f} DKK\n"
                        context_prefix += f"  - Median revenue per item: {cat_median_rev:,.0f} DKK\n"

                        # Show examples
                        if category == 'Dog':
                            examples = cat_items.nsmallest(10, 'revenue') if 'revenue' in cat_items.columns else cat_items.head(10)
                            context_prefix += f"  - Worst 10 examples:\n"
                        else:
                            examples = cat_items.nlargest(10, 'revenue') if 'revenue' in cat_items.columns else cat_items.head(10)
                            context_prefix += f"  - Top 10 examples:\n"

                        for idx, item in examples.iterrows():
                            name = item.get('item_name', 'Unknown')
                            revenue = item.get('revenue', 0)
                            orders = item.get('order_count', 0)
                            context_prefix += f"    • {name}: {revenue:,.0f} DKK, {orders} orders\n"

            # LOW PERFORMERS ANALYSIS (for "what should I remove" questions)
            if 'revenue' in df.columns and 'order_count' in df.columns:
                context_prefix += f"\nLOW PERFORMERS (items with lowest revenue AND low orders):\n"
                # Items in bottom 25% of both revenue and orders
                low_rev_threshold = df['revenue'].quantile(0.25)
                low_order_threshold = df['order_count'].quantile(0.25)
                low_performers = df[(df['revenue'] <= low_rev_threshold) & (df['order_count'] <= low_order_threshold)]
                context_prefix += f"  - {len(low_performers):,} items below 25th percentile in BOTH revenue and orders\n"
                if len(low_performers) > 0:
                    context_prefix += f"  - These items average {low_performers['revenue'].mean():,.0f} DKK revenue\n"
                    context_prefix += f"  - These items average {low_performers['order_count'].mean():.0f} orders\n"
                    context_prefix += f"  - Examples (worst 15):\n"
                    for idx, item in low_performers.nsmallest(15, 'revenue').iterrows():
                        name = item.get('item_name', 'Unknown')
                        revenue = item.get('revenue', 0)
                        orders = item.get('order_count', 0)
                        context_prefix += f"    • {name}: {revenue:,.0f} DKK, {orders} orders\n"

            # Timeline data summary if available
            if self.timeline_df is not None and restaurant_id and 'place_id' in self.timeline_df.columns:
                timeline_data = self.timeline_df[self.timeline_df['place_id'] == restaurant_id]
                if len(timeline_data) > 0:
                    context_prefix += f"\n\nTIMELINE DATA: {len(timeline_data):,} orders with timestamps\n"

            context_prefix += f"\n\nYOU HAVE COMPLETE STATISTICS FOR ALL {total_items:,} ITEMS. Use these stats to answer analytical questions.\n"

            # Build the full prompt
            full_prompt = self.data_context + context_prefix + f"\n\nUSER QUESTION: {question}\n\nProvide a helpful, data-driven answer:"

            # Generate response using Groq API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a menu engineering expert helping restaurants optimize their menus. CRITICAL: You have REAL DATA provided in the context. NEVER make up SQL queries, NEVER invent products or numbers. ONLY use the exact items, counts, and statistics shown in the context above. If timeline data shows specific items like '12:00 [Pizza (78), Pasta (54)]', use those EXACT names and counts. Always use ONLY the data provided for the specific restaurant being analyzed. Never mix data from different restaurants. If you don't have specific information, say 'I don't have that specific data' instead of guessing."
                    },
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            # Get the response text
            answer = response.choices[0].message.content

            return answer

        except Exception as e:
            logger.error(f"Error in AI assistant: {e}")
            return f"Error: {str(e)}"

    def get_restaurant_summary(self, restaurant_id: int) -> dict:
        """Get a quick summary of a restaurant's menu"""
        if self.classification_df is None:
            return None

        df = self.classification_df

        if 'place_id' in df.columns:
            df = df[df['place_id'] == restaurant_id]

        if len(df) == 0:
            return None

        summary = {
            'total_items': len(df),
        }

        if 'revenue' in df.columns:
            summary['total_revenue'] = float(df['revenue'].sum())
        if 'order_count' in df.columns:
            summary['total_orders'] = int(df['order_count'].sum())
        if 'category' in df.columns:
            summary['categories'] = df['category'].value_counts().to_dict()
        if 'revenue' in df.columns:
            top_item = df.nlargest(1, 'revenue').iloc[0].to_dict()
            summary['top_item'] = top_item

        return summary
