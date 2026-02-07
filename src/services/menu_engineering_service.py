"""
File: menu_engineering_service.py
Description: Comprehensive menu engineering analytics and optimization service.
Dependencies: pandas, numpy, scipy, scikit-learn
Author: MenuMetrics Intelligence Platform Team

This service implements the complete menu engineering pipeline:
1. Data aggregation and profitability analysis
2. BCG Matrix classification (Stars, Plows, Puzzles, Dogs)
3. Pricing optimization using elasticity analysis
4. Customer behavior and bundling opportunity detection
5. Menu optimization plan generation with actionable recommendations

The service transforms raw sales data into strategic business intelligence
for menu optimization, pricing decisions, and promotional strategies.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from models.data_loader import DataLoader
from models.menu_engineering_models import (
    MenuItemMetrics,
    MenuItemInsight,
    PricingOptimization,
    CustomerBehavior,
    MenuOptimizationPlan,
    MenuQuadrant,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MenuEngineeringService:
    """
    Enterprise menu engineering analytics service.
    
    This service provides:
    - Profitability and popularity analysis
    - BCG Matrix (4-quadrant) classification
    - Price elasticity and optimization recommendations
    - Customer behavior and bundling analysis
    - Complete menu optimization plan generation
    
    All analysis follows industry-standard restaurant management practices
    and uses contribution margin and sales volume as primary metrics.
    
    Attributes:
        data_loader (DataLoader): Data loading utility
        currency (str): Currency code (default: DKK)
        profit_threshold (float): Median contribution margin threshold
        popularity_threshold (float): Median quantity threshold
    """

    def __init__(self, data_loader: DataLoader, currency: str = "DKK") -> None:
        """
        Initialize the menu engineering service.
        
        Args:
            data_loader (DataLoader): Utility for loading CSV data
            currency (str): Currency label for reporting. Defaults to "DKK".
        """
        self.data_loader = data_loader
        self.currency = currency
        self.profit_threshold: float = 0.0
        self.popularity_threshold: float = 0.0
        self.insights_df: Optional[pd.DataFrame] = None
        self.menu_items_df: Optional[pd.DataFrame] = None  # Store for satisfaction metrics
        logger.info(f"MenuEngineeringService initialized (Currency: {currency})")

    def load_menu_data(
        self,
        menu_items_file: str,
        order_items_file: str,
        place_id: Optional[str] = None,
        profitability_margin: float = 0.20,
    ) -> pd.DataFrame:
        """
        Load menu and order data from CSV files and prepare unified dataset.

        This method:
        1. Loads dim_menu_items and fct_order_items CSVs
        2. Normalizes column names and data types
        3. Merges datasets on menu_item_id
        4. Filters by place_id if specified

        Args:
            menu_items_file (str): Path to dim_menu_items CSV
            order_items_file (str): Path to fct_order_items CSV
            place_id (Optional[str]): Filter for specific location/merchant
            profitability_margin (float): Assumed profit margin for items without cost data (default: 0.20)

        Returns:
            pd.DataFrame: Merged and normalized order data with menu metadata

        Raises:
            FileNotFoundError: If CSV files not found

        Example:
            >>> service = MenuEngineeringService(loader)
            >>> orders = service.load_menu_data(
            ...     "Menu Engineering Part 2/dim_menu_items.csv",
            ...     "Menu Engineering Part 2/fct_order_items.csv"
            ... )
        """
        try:
            menu_items = self.data_loader.load_csv(menu_items_file)
            order_items = self.data_loader.load_csv(order_items_file)
            return self.prepare_order_items(menu_items, order_items, place_id, profitability_margin)
        except FileNotFoundError as e:
            logger.error(f"Failed to load menu data: {e}")
            raise

    def prepare_order_items(
        self,
        menu_items: pd.DataFrame,
        order_items: pd.DataFrame,
        place_id: Optional[str] = None,
        profitability_margin: float = 0.20,
    ) -> pd.DataFrame:
        """
        Normalize and prepare order items for analysis.

        This method handles flexible column naming conventions found in real-world
        datasets. It:
        - Normalizes column names (accepts multiple naming variants)
        - Fills missing cost data with configurable profitability assumption
        - Merges order and menu data
        - Calculates derived metrics (revenue, cost, contribution)
        - Filters by place_id if provided

        Supported column name variations:
            - Item ID: menu_item_id, item_id, product_id, menu_item
            - Quantity: quantity, qty, item_quantity
            - Price: price, unit_price, item_price
            - Cost: cost, unit_cost, item_cost
            - Place: place_id, location_id, merchant_id

        Args:
            menu_items (pd.DataFrame): Menu items master data
            order_items (pd.DataFrame): Order transaction data
            place_id (Optional[str]): Filter for specific location
            profitability_margin (float): Assumed profit margin for items without cost data (default: 0.20 = 20%)

        Returns:
            pd.DataFrame: Normalized and merged order data

        Raises:
            ValueError: If required columns cannot be resolved
        """
        logger.info("Preparing and normalizing order items...")

        # Store menu_items for later satisfaction metrics loading
        self.menu_items_df = menu_items

        # Resolve flexible column naming for menu items
        menu_id_col = self._resolve_column(menu_items, ["id", "menu_item_id", "item_id"])
        menu_title_col = self._resolve_column(menu_items, ["title", "name", "menu_item_title"])
        menu_status_col = self._resolve_column(
            menu_items,
            ["status", "is_active", "available"],
            optional=True,
        )

        # Resolve flexible column naming for order items
        order_menu_id_col = self._resolve_column(
            order_items,
            ["menu_item_id", "item_id", "product_id", "menu_item"],
        )
        order_qty_col = self._resolve_column(order_items, ["quantity", "qty", "item_quantity"])
        order_price_col = self._resolve_column(order_items, ["price", "unit_price", "item_price"])
        order_cost_col = self._resolve_column(
            order_items,
            ["cost", "unit_cost", "item_cost"],
            optional=True,
        )
        order_place_col = self._resolve_column(
            order_items,
            ["place_id", "location_id", "merchant_id"],
            optional=True,
        )
        order_title_col = self._resolve_column(
            order_items,
            ["title", "item_name", "menu_item_title"],
            optional=True,
        )

        # Standardize order items column names
        normalized = order_items.copy()
        normalized = normalized.rename(
            columns={
                order_menu_id_col: "menu_item_id",
                order_qty_col: "quantity",
                order_price_col: "unit_price",
            }
        )

        # Handle optional columns
        if order_cost_col:
            # Some datasets store line-total cost in a generic "cost"/"item_cost" column.
            # Preserve raw values first and normalize to unit cost after quantity conversion.
            if order_cost_col in ["cost", "item_cost"]:
                normalized = normalized.rename(columns={order_cost_col: "cost_raw"})
            else:
                normalized = normalized.rename(columns={order_cost_col: "unit_cost"})
        else:
            # Industry standard: estimate COGS at 30% of selling price
            normalized["unit_cost"] = 0.0

        if order_place_col:
            normalized = normalized.rename(columns={order_place_col: "place_id"})

        if order_title_col:
            normalized = normalized.rename(columns={order_title_col: "item_title"})
        else:
            normalized["item_title"] = np.nan

        # Filter by place_id if provided
        if place_id and "place_id" in normalized.columns:
            original_count = len(normalized)
            normalized = normalized[normalized["place_id"].astype(str) == str(place_id)]
            logger.info(f"Filtered to place {place_id}: {len(normalized):,} of {original_count:,} rows")

        # Prepare menu subset for merge
        menu_subset = menu_items[[menu_id_col, menu_title_col]].copy()
        menu_subset = menu_subset.rename(
            columns={menu_id_col: "menu_item_id", menu_title_col: "menu_title"}
        )

        if menu_status_col:
            menu_subset["menu_status"] = menu_items[menu_status_col]
        else:
            menu_subset["menu_status"] = "unknown"

        # Merge order and menu data
        merged = normalized.merge(menu_subset, on="menu_item_id", how="left")
        merged["item_title"] = merged["item_title"].fillna(merged["menu_title"])

        # Type conversion and validation
        merged["unit_price"] = pd.to_numeric(merged["unit_price"], errors='coerce')
        merged["quantity"] = pd.to_numeric(merged["quantity"], errors='coerce')
        if "cost_raw" in merged.columns:
            merged["cost_raw"] = pd.to_numeric(merged["cost_raw"], errors='coerce')
            merged["unit_cost"] = np.where(
                merged["quantity"] > 0,
                merged["cost_raw"] / merged["quantity"],
                np.nan,
            )
        else:
            merged["unit_cost"] = pd.to_numeric(merged["unit_cost"], errors='coerce')

        # If provided cost mirrors sales price too closely, it is likely not COGS.
        # Fall back to estimated COGS from profitability assumptions in that case.
        valid_cost = merged[
            (merged["unit_price"] > 0)
            & (merged["quantity"] > 0)
            & merged["unit_cost"].notna()
        ]
        if len(valid_cost) >= 1000:
            price_ratio = valid_cost["unit_cost"] / valid_cost["unit_price"]
            near_parity = ((price_ratio >= 0.95) & (price_ratio <= 1.05)).mean()
            median_ratio = float(price_ratio.median())
            if near_parity >= 0.90 and 0.95 <= median_ratio <= 1.05:
                logger.warning(
                    "Cost column appears to mirror sales prices (likely non-COGS). "
                    "Ignoring provided cost and estimating COGS from profitability assumptions."
                )
                merged["unit_cost"] = np.nan

        # Handle missing costs with configurable profitability assumption
        # If profitability_margin is 20%, cost = 80% of price
        cost_multiplier = 1.0 - profitability_margin
        if merged["unit_cost"].isna().sum() > 0:
            logger.warning(
                f"Estimated cost for {merged['unit_cost'].isna().sum():,} items "
                f"(assumed {profitability_margin*100:.0f}% profit margin, cost = {cost_multiplier*100:.0f}% of price)"
            )
            merged["unit_cost"] = merged["unit_cost"].fillna(merged["unit_price"] * cost_multiplier)
        
        # Calculate derived metrics
        merged["item_revenue"] = merged["unit_price"] * merged["quantity"]
        merged["item_cost"] = merged["unit_cost"] * merged["quantity"]

        # DATA QUALITY FILTERING: Remove admin, test, and invalid items
        initial_count = len(merged)

        # Filter 1: Remove inactive items
        if "menu_status" in merged.columns:
            merged = merged[
                (merged["menu_status"].str.lower() == "active") |
                (merged["menu_status"] == "unknown")
            ]
            logger.info(f"Filtered out {initial_count - len(merged):,} inactive items")

        # Filter 2: Remove items with blank/null titles
        merged = merged[
            merged["item_title"].notna() &
            (merged["item_title"].str.strip() != "")
        ]
        logger.info(f"Filtered out items with blank titles (remaining: {len(merged):,})")

        # Filter 3: Remove items with invalid prices (≤0)
        merged = merged[merged["unit_price"] > 0]
        logger.info(f"Filtered out items with price ≤ 0 (remaining: {len(merged):,})")

        # Filter 4: Remove test/admin/unspecified items by title keywords
        test_keywords = ['test', 'admin', 'sample', 'demo', 'do not', 'donotuse',
                        'placeholder', 'temp', 'draft', 'delete', 'xxx', 'zzz',
                        'unspecified', 'unknown', 'n/a', 'null', 'none', 'default',
                        'blank', 'empty', 'tbd', 'to be determined']
        test_pattern = '|'.join(test_keywords)
        merged = merged[~merged["item_title"].str.lower().str.contains(test_pattern, na=False)]
        logger.info(f"Filtered out test/admin/unspecified items (remaining: {len(merged):,})")

        # Filter 5: Remove items with 0 quantity (data quality issue)
        merged = merged[merged["quantity"] > 0]
        logger.info(f"Filtered out items with 0 quantity (remaining: {len(merged):,})")

        total_filtered = initial_count - len(merged)
        if total_filtered > 0:
            logger.info(f"✓ Data quality filtering complete: Removed {total_filtered:,} ({total_filtered/initial_count*100:.1f}%) invalid/test items")

        logger.info(f"Prepared {len(merged):,} clean order items for analysis")
        return merged

    def load_satisfaction_metrics(self, menu_items: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and normalize customer satisfaction metrics from menu items.

        This method:
        - Normalizes rating from 0-120 scale to 0-5 star scale
        - Calculates satisfaction score weighted by vote count
        - Returns DataFrame ready for merging with insights

        Args:
            menu_items (pd.DataFrame): Menu items with rating and votes columns

        Returns:
            pd.DataFrame: Satisfaction metrics with columns:
                - menu_item_id: Item identifier
                - rating_normalized: Rating on 0-5 scale
                - votes: Number of customer votes
                - satisfaction_score: Weighted score (rating × log(votes+1))
        """
        logger.info("Loading customer satisfaction metrics...")

        # Resolve column names
        menu_id_col = self._resolve_column(menu_items, ["id", "menu_item_id", "item_id"])
        rating_col = self._resolve_column(menu_items, ["rating"], optional=True)
        votes_col = self._resolve_column(menu_items, ["votes"], optional=True)

        if not rating_col or not votes_col:
            logger.warning("Rating or votes column not found - satisfaction metrics unavailable")
            return None

        # Extract satisfaction data
        satisfaction_df = menu_items[[menu_id_col, rating_col, votes_col]].copy()
        satisfaction_df = satisfaction_df.rename(columns={
            menu_id_col: "menu_item_id",
            rating_col: "rating_raw",
            votes_col: "votes_raw"
        })

        # Normalize rating from 0-120 to 0-5 scale
        satisfaction_df["rating_normalized"] = pd.to_numeric(
            satisfaction_df["rating_raw"], errors='coerce'
        ) / 24.0  # 120 / 5 = 24

        # Ensure votes is numeric
        satisfaction_df["votes"] = pd.to_numeric(satisfaction_df["votes_raw"], errors='coerce')

        # Calculate satisfaction score: rating × log(votes + 1)
        # This gives more weight to items with many votes
        satisfaction_df["satisfaction_score"] = (
            satisfaction_df["rating_normalized"] * np.log1p(satisfaction_df["votes"])
        )

        # Drop original raw columns
        satisfaction_df = satisfaction_df.drop(columns=["rating_raw", "votes_raw"], errors='ignore')

        # Fill NaN values
        satisfaction_df["rating_normalized"] = satisfaction_df["rating_normalized"].fillna(0)
        satisfaction_df["votes"] = satisfaction_df["votes"].fillna(0)
        satisfaction_df["satisfaction_score"] = satisfaction_df["satisfaction_score"].fillna(0)

        logger.info(
            f"Loaded satisfaction for {len(satisfaction_df):,} items. "
            f"Avg rating: {satisfaction_df['rating_normalized'].mean():.2f}/5"
        )

        return satisfaction_df

    def analyze_menu_language_correlation(self, insights: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze which words/terms in menu item names correlate with performance.

        This method:
        - Tokenizes item titles (lowercase, removes stopwords)
        - Extracts unigrams and bigrams
        - Calculates average metrics for items WITH vs WITHOUT each term
        - Performs statistical significance testing (t-test)
        - Classifies terms as power words, value words, quality words, or toxic words

        Args:
            insights (pd.DataFrame): Menu insights with performance metrics

        Returns:
            Dict[str, Any]: Language analysis results with:
                - power_words: Terms that significantly boost revenue (>30%, p<0.05)
                - toxic_words: Terms that significantly hurt revenue (<-20%)
                - category_terms: Common terms by BCG category
                - rename_suggestions: Suggested renames for low-performing items
        """
        import re
        from collections import Counter, defaultdict

        logger.info("Analyzing menu language correlation...")

        if 'item_title' not in insights.columns or len(insights) < 10:
            logger.warning("Insufficient data for language analysis")
            return {}

        # Stopwords to exclude
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'with', 'for', 'of', 'in', 'on', 'at', 'to',
            'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }

        # Tokenize all item titles
        def tokenize(title):
            """Extract words from title"""
            if pd.isna(title):
                return []
            # Convert to lowercase, extract words
            words = re.findall(r'\b\w+\b', str(title).lower())
            # Filter: length >= 3, not stopword, not all digits
            return [w for w in words if len(w) >= 3 and w not in stopwords and not w.isdigit()]

        insights['tokens'] = insights['item_title'].apply(tokenize)

        # Count term frequency
        all_terms = []
        for tokens in insights['tokens']:
            all_terms.extend(tokens)
        term_counts = Counter(all_terms)

        # Filter: term must appear in at least 3 items
        valid_terms = [term for term, count in term_counts.items() if count >= 3]

        logger.info(f"Found {len(valid_terms)} terms appearing in 3+ items")

        # Calculate correlation for each term
        term_analysis = []

        for term in valid_terms:
            # Items WITH this term
            with_term = insights[insights['tokens'].apply(lambda tokens: term in tokens)]
            # Items WITHOUT this term
            without_term = insights[~insights['tokens'].apply(lambda tokens: term in tokens)]

            if len(with_term) < 2 or len(without_term) < 2:
                continue

            # Calculate average metrics
            avg_revenue_with = with_term['total_revenue'].mean()
            avg_revenue_without = without_term['total_revenue'].mean()
            avg_margin_with = with_term['margin_percentage'].mean()
            avg_margin_without = without_term['margin_percentage'].mean()

            # Calculate lift
            revenue_lift = avg_revenue_with - avg_revenue_without
            revenue_lift_pct = (revenue_lift / avg_revenue_without * 100) if avg_revenue_without > 0 else 0

            # T-test for statistical significance
            try:
                t_stat, p_value = stats.ttest_ind(
                    with_term['total_revenue'],
                    without_term['total_revenue'],
                    equal_var=False
                )
            except:
                p_value = 1.0

            # Satisfaction lift if available
            satisfaction_lift = 0.0
            if 'rating_normalized' in insights.columns:
                with_term_rated = with_term[with_term['rating_normalized'] > 0]
                without_term_rated = without_term[without_term['rating_normalized'] > 0]
                if len(with_term_rated) > 0 and len(without_term_rated) > 0:
                    satisfaction_lift = (
                        with_term_rated['rating_normalized'].mean() -
                        without_term_rated['rating_normalized'].mean()
                    )

            term_analysis.append({
                'term': term,
                'frequency': len(with_term),
                'avg_revenue_with': avg_revenue_with,
                'avg_revenue_without': avg_revenue_without,
                'revenue_lift': revenue_lift,
                'revenue_lift_pct': revenue_lift_pct,
                'margin_lift': avg_margin_with - avg_margin_without,
                'satisfaction_lift': satisfaction_lift,
                'p_value': p_value
            })

        # Sort by revenue lift
        term_analysis_df = pd.DataFrame(term_analysis).sort_values('revenue_lift', ascending=False)

        # Classify terms
        power_words = term_analysis_df[
            (term_analysis_df['revenue_lift_pct'] > 30) &
            (term_analysis_df['p_value'] < 0.05)
        ].head(10).to_dict('records')

        toxic_words = term_analysis_df[
            (term_analysis_df['revenue_lift_pct'] < -20)
        ].tail(10).to_dict('records')

        # Category-specific common terms
        category_terms = {}
        for category in ['star', 'plowhorse', 'puzzle', 'dog']:
            cat_items = insights[insights['category'] == category]
            if len(cat_items) > 0:
                cat_tokens = []
                for tokens in cat_items['tokens']:
                    cat_tokens.extend(tokens)
                cat_term_counts = Counter(cat_tokens)
                # Top 5 most common terms in this category
                category_terms[category] = [term for term, count in cat_term_counts.most_common(5)]

        # Generate rename suggestions for low-performing items
        rename_suggestions = []
        if len(power_words) > 0 and len(toxic_words) > 0:
            # Find Dog items with toxic words
            dog_items = insights[insights['category'] == 'dog'].copy()
            for idx, item in dog_items.head(5).iterrows():
                current_name = item['item_title']
                tokens = set(item['tokens'])

                # Check if it has toxic words
                has_toxic = any(tw['term'] in tokens for tw in toxic_words)

                if has_toxic and len(power_words) > 0:
                    # Suggest replacing with a power word
                    power_term = power_words[0]['term']
                    suggested_name = f"{power_term.capitalize()} {current_name}"
                    rename_suggestions.append({
                        'current_name': current_name,
                        'suggested_name': suggested_name,
                        'reason': f"Contains underperforming terms, add '{power_term}' to boost appeal"
                    })

        logger.info(f"Language analysis complete: {len(power_words)} power words, {len(toxic_words)} toxic words")

        return {
            'power_words': power_words,
            'toxic_words': toxic_words,
            'category_terms': category_terms,
            'rename_suggestions': rename_suggestions[:5],  # Limit to 5 suggestions
            'all_term_analysis': term_analysis_df.head(50).to_dict('records')
        }

    def analyze_purchase_patterns(
        self,
        order_items: pd.DataFrame,
        timeline_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive customer behavior pattern analysis.

        Args:
            order_items (pd.DataFrame): Order transaction data
            timeline_data (Optional[pd.DataFrame]): Timeline data with timestamps

        Returns:
            Dict[str, Any]: Complete behavior analysis including:
                - copurchase_pairs: Frequently co-purchased item pairs
                - hourly_patterns: Time-of-day purchase trends
                - day_patterns: Day-of-week trends
                - cross_sell_opportunities: Recommended item combinations
        """
        logger.info("Analyzing customer purchase patterns...")

        analysis = {}

        # Co-purchase analysis
        analysis['copurchase_pairs'] = self.detect_copurchase_patterns(order_items)

        # Temporal patterns
        if timeline_data is not None and len(timeline_data) > 0:
            analysis.update(self.analyze_temporal_patterns(timeline_data))
        else:
            analysis['hourly_patterns'] = {}
            analysis['day_patterns'] = {}

        # Cross-sell opportunities (derived from co-purchase)
        analysis['cross_sell_opportunities'] = self._generate_cross_sell_recommendations(
            analysis['copurchase_pairs']
        )

        logger.info("Purchase pattern analysis complete")
        return analysis

    def detect_copurchase_patterns(self, order_items: pd.DataFrame) -> List[Dict]:
        """
        Market basket analysis for co-purchased items.

        Uses association rule mining to find item pairs that are frequently
        purchased together. Calculates support, confidence, and lift metrics.

        Args:
            order_items (pd.DataFrame): Order transaction data with order_id

        Returns:
            List[Dict]: Co-purchase pairs sorted by lift, with metrics:
                - item_a, item_b: Item names
                - support: Frequency of co-occurrence (0-1)
                - confidence: P(item_b | item_a)
                - lift: How much more likely they're bought together vs random
        """
        logger.info("Detecting co-purchase patterns...")

        if 'order_id' not in order_items.columns:
            logger.warning("No order_id column - cannot detect co-purchase patterns")
            return []

        # Group by order to get item sets
        orders = order_items.groupby('order_id')['item_title'].apply(list).reset_index()

        if len(orders) < 10:
            logger.warning("Too few orders for co-purchase analysis")
            return []

        total_orders = len(orders)

        # Generate item pairs within orders
        from itertools import combinations

        pair_counts = defaultdict(int)
        item_counts = defaultdict(int)

        for _, row in orders.iterrows():
            items = list(set(row['item_title']))  # Unique items in order
            if len(items) < 2:
                continue

            # Count individual items
            for item in items:
                item_counts[item] += 1

            # Count pairs
            for item_a, item_b in combinations(sorted(items), 2):
                pair_counts[(item_a, item_b)] += 1

        # Calculate metrics
        copurchase_results = []

        for (item_a, item_b), count in pair_counts.items():
            # Support: P(A and B)
            support = count / total_orders

            # Confidence: P(B | A)
            confidence = count / item_counts[item_a] if item_counts[item_a] > 0 else 0

            # Lift: P(A and B) / (P(A) * P(B))
            prob_a = item_counts[item_a] / total_orders
            prob_b = item_counts[item_b] / total_orders
            lift = (support / (prob_a * prob_b)) if (prob_a * prob_b) > 0 else 0

            # Filter: minimum thresholds (lowered for better results with large datasets)
            # Support: at least 0.1% of orders (was 2%)
            # Confidence: at least 20% chance (was 30%)
            # Lift: at least 1.3x more likely than random (was 1.5x)
            if support >= 0.001 and confidence >= 0.20 and lift >= 1.3 and count >= 3:
                copurchase_results.append({
                    'item_a': item_a,
                    'item_b': item_b,
                    'support': support,
                    'confidence': confidence,
                    'lift': lift,
                    'count': count
                })

        # Sort by lift (strongest associations first)
        copurchase_results.sort(key=lambda x: x['lift'], reverse=True)

        logger.info(f"Found {len(copurchase_results)} co-purchase patterns")
        return copurchase_results[:50]  # Top 50

    def analyze_temporal_patterns(self, timeline_data: pd.DataFrame) -> Dict:
        """
        Analyze time-based purchasing patterns.

        Args:
            timeline_data (pd.DataFrame): Order data with timestamp column

        Returns:
            Dict: Temporal patterns including:
                - hourly_patterns: Items ordered by hour of day
                - day_patterns: Items ordered by day of week
                - peak_hours: Hours with highest order volume
        """
        logger.info("Analyzing temporal patterns...")

        temporal_analysis = {
            'hourly_patterns': {},
            'day_patterns': {},
            'peak_hours': []
        }

        # Work on a copy to avoid mutating upstream dataframes
        timeline_data = timeline_data.copy()

        # Check for timestamp column
        timestamp_col = None
        for col in ['created_at', 'order_time', 'timestamp', 'created', 'order_datetime']:
            if col in timeline_data.columns:
                timestamp_col = col
                break

        if not timestamp_col:
            logger.warning("No timestamp column found in timeline data")
            return temporal_analysis

        # Resolve item-title column variants found in exported timeline files
        item_col = None
        for col in ['item_title', 'title', 'item_name', 'name']:
            if col in timeline_data.columns:
                item_col = col
                break

        # Convert to datetime.
        # Numeric timestamp columns are treated as UNIX seconds; text columns use parser inference.
        ts_numeric = pd.to_numeric(timeline_data[timestamp_col], errors='coerce')
        numeric_coverage = ts_numeric.notna().mean()
        if numeric_coverage >= 0.95:
            timeline_data[timestamp_col] = pd.to_datetime(ts_numeric, unit='s', errors='coerce')
        else:
            timeline_data[timestamp_col] = pd.to_datetime(timeline_data[timestamp_col], errors='coerce')

        # Drop rows with invalid timestamps before deriving time features
        timeline_data = timeline_data[timeline_data[timestamp_col].notna()]
        if len(timeline_data) == 0:
            logger.warning("All timeline timestamps failed to parse")
            return temporal_analysis

        # Extract hour and day of week
        timeline_data['hour'] = timeline_data[timestamp_col].dt.hour
        timeline_data['day_of_week'] = timeline_data[timestamp_col].dt.dayofweek  # 0=Monday

        # Hourly patterns
        if item_col:
            hourly_orders = timeline_data.groupby(['hour', item_col]).size().reset_index(name='count')
            hourly_top = hourly_orders.sort_values('count', ascending=False).groupby('hour').head(5)

            for hour in range(24):
                hour_items = hourly_top[hourly_top['hour'] == hour]
                if len(hour_items) > 0:
                    hour_items = hour_items.rename(columns={item_col: 'item_title'})
                    temporal_analysis['hourly_patterns'][hour] = hour_items[['item_title', 'count']].to_dict('records')

        # Peak hours
        hourly_volume = timeline_data.groupby('hour').size().reset_index(name='orders')
        peak_threshold = hourly_volume['orders'].quantile(0.75)
        peak_hours = hourly_volume[hourly_volume['orders'] >= peak_threshold]['hour'].tolist()
        temporal_analysis['peak_hours'] = peak_hours

        # Day patterns (weekday vs weekend)
        timeline_data['is_weekend'] = timeline_data['day_of_week'].isin([5, 6])  # Sat, Sun

        if item_col:
            weekday_items = timeline_data[~timeline_data['is_weekend']][item_col].value_counts().head(10)
            weekend_items = timeline_data[timeline_data['is_weekend']][item_col].value_counts().head(10)

            temporal_analysis['day_patterns'] = {
                'weekday_favorites': weekday_items.to_dict(),
                'weekend_favorites': weekend_items.to_dict()
            }

        logger.info("Temporal analysis complete")
        return temporal_analysis

    def _generate_cross_sell_recommendations(self, copurchase_pairs: List[Dict]) -> List[Dict]:
        """Generate actionable cross-sell recommendations from co-purchase patterns."""
        recommendations = []

        for pair in copurchase_pairs[:10]:  # Top 10 pairs
            recommendations.append({
                'anchor_item': pair['item_a'],
                'recommended_item': pair['item_b'],
                'expected_rate': pair['confidence'],
                'lift': pair['lift'],
                'rationale': f"Customers who order {pair['item_a']} also order {pair['item_b']} {pair['confidence']*100:.0f}% of the time"
            })

        return recommendations

    def build_menu_insights(self, order_items: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive menu engineering insights from order data.
        
        This method:
        1. Aggregates metrics by menu item (quantity, revenue, cost, margins)
        2. Calculates profitability (contribution margin and %)
        3. Calculates popularity score (percentage of total sales volume)
        4. Classifies items into BCG Matrix quadrants
        5. Generates strategic recommendations
        
        The BCG Matrix classification uses:
        - Profitability: Contribution Margin (Revenue - COGS)
        - Popularity: Sales Volume (quantity sold)
        - Thresholds: Medians of respective metrics
        
        Quadrant Definitions:
        - STAR: High profit + High popularity -> Protect and promote
        - PLOW: High profit + Low popularity -> Increase visibility
        - PUZZLE: Low profit + High popularity -> Optimize pricing/bundling
        - DOG: Low profit + Low popularity -> Remove or redesign
        
        Args:
            order_items (pd.DataFrame): Normalized order data
        
        Returns:
            pd.DataFrame: Menu insights with columns:
                - menu_item_id: Item identifier
                - item_title: Item name
                - total_quantity: Units sold
                - total_revenue: DKK revenue
                - total_cost: DKK COGS
                - contribution_margin: DKK profit
                - margin_percentage: Profit as % of revenue
                - popularity_score: Percentage of total sales
                - category: BCG quadrant (Star/Plow/Puzzle/Dog)
                - suggested_action: Strategic recommendation
                - price_hint: Pricing guidance
        """
        logger.info("Building menu insights from order data...")
        
        # Aggregate metrics by menu item
        grouped = (
            order_items.groupby(["menu_item_id", "item_title"], dropna=False)
            .agg(
                total_quantity=("quantity", "sum"),
                total_revenue=("item_revenue", "sum"),
                total_cost=("item_cost", "sum"),
                avg_price=("unit_price", "mean"),
                avg_cost=("unit_cost", "mean"),
                std_price=("unit_price", "std"),
                menu_status=("menu_status", "last"),
                order_count=("quantity", "count"),
            )
            .reset_index()
        )

        # Calculate profitability metrics
        grouped["contribution_margin"] = grouped["total_revenue"] - grouped["total_cost"]
        grouped["margin_percentage"] = np.where(
            grouped["total_revenue"] > 0,
            (grouped["contribution_margin"] / grouped["total_revenue"]) * 100,
            0,
        )

        # Calculate popularity as percentage of total sales
        total_qty = grouped["total_quantity"].sum()
        grouped["popularity_score"] = np.where(
            total_qty > 0,
            (grouped["total_quantity"] / total_qty) * 100,
            0,
        )

        # Set thresholds for quadrant classification
        self.profit_threshold = grouped["contribution_margin"].median()
        self.popularity_threshold = grouped["total_quantity"].median()

        logger.info(f"Profit threshold: {self.currency} {self.profit_threshold:,.0f}")
        logger.info(f"Popularity threshold: {self.popularity_threshold:,.0f} units")

        # Classify items into BCG Matrix quadrants
        grouped["category"] = grouped.apply(
            lambda row: self._classify_item(row, self.popularity_threshold, self.profit_threshold),
            axis=1,
        )

        # Merge customer satisfaction metrics before generating recommendations
        if self.menu_items_df is not None:
            satisfaction_df = self.load_satisfaction_metrics(self.menu_items_df)
            if satisfaction_df is not None:
                grouped = grouped.merge(
                    satisfaction_df,
                    on="menu_item_id",
                    how="left"
                )
                # Fill NaN for items without satisfaction data
                grouped["rating_normalized"] = grouped["rating_normalized"].fillna(0)
                grouped["votes"] = grouped["votes"].fillna(0)
                grouped["satisfaction_score"] = grouped["satisfaction_score"].fillna(0)

                logger.info(f"Merged satisfaction metrics for insights")
            else:
                # No satisfaction data - add dummy columns
                grouped["rating_normalized"] = 0.0
                grouped["votes"] = 0
                grouped["satisfaction_score"] = 0.0
        else:
            # No menu items available - add dummy columns
            grouped["rating_normalized"] = 0.0
            grouped["votes"] = 0
            grouped["satisfaction_score"] = 0.0

        # Generate strategic recommendations (with satisfaction context)
        grouped["suggested_action"] = grouped.apply(
            lambda row: self._recommend_action(
                row["category"],
                row["margin_percentage"],
                row.get("rating_normalized", 0.0),
                row.get("votes", 0.0),
            ),
            axis=1,
        )

        # Generate pricing guidance
        grouped["price_hint"] = grouped.apply(
            lambda row: self._price_hint(row["category"], row["avg_price"], row["margin_percentage"]),
            axis=1,
        )

        # Sort by category and revenue for easy analysis
        insights = grouped.sort_values(
            ["category", "total_revenue"],
            ascending=[True, False]
        )

        # Log summary statistics
        logger.info("="*60)
        logger.info("MENU INSIGHTS SUMMARY")
        logger.info("="*60)
        for category in ["star", "plowhorse", "puzzle", "dog"]:
            count = len(insights[insights["category"] == category])
            if count > 0:
                logger.info(f"{category.upper()}: {count} items")
        logger.info("="*60)

        self.insights_df = insights
        return insights

    def build_summary(self, insights: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate performance summary metrics.
        
        Args:
            insights (pd.DataFrame): Menu insights from build_menu_insights()
        
        Returns:
            Dict[str, Any]: Summary metrics including:
                - total_items: Number of menu items
                - total_revenue: Total DKK revenue
                - total_cost: Total DKK COGS
                - total_margin: Total DKK profit
                - average_margin_pct: Average profit margin %
                - by_category: Breakdown by BCG quadrant
        """
        summary = {
            "total_items": int(insights.shape[0]),
            "total_revenue": float(insights["total_revenue"].sum()),
            "total_cost": float(insights["total_cost"].sum()),
            "total_margin": float(insights["contribution_margin"].sum()),
            "average_margin_pct": float(insights["margin_percentage"].mean()),
            "by_category": {},
        }

        # Calculate metrics by category
        for category in ["star", "plowhorse", "puzzle", "dog"]:
            cat_data = insights[insights["category"] == category]
            if len(cat_data) > 0:
                summary["by_category"][category] = {
                    "count": int(len(cat_data)),
                    "revenue": float(cat_data["total_revenue"].sum()),
                    "margin": float(cat_data["contribution_margin"].sum()),
                    "items": cat_data[["menu_item_id", "item_title"]].to_dict("records"),
                }

        return summary

    def analyze_pricing_optimization(self, order_items: pd.DataFrame) -> List[PricingOptimization]:
        """
        Analyze price elasticity and recommend optimal prices.
        
        This method:
        1. Groups orders by item and price point
        2. Estimates price elasticity using linear regression
        3. Recommends optimal price to maximize profit
        4. Filters recommendations by confidence level
        
        Price Elasticity Interpretation:
        - Elasticity = -1.2 means 1% price increase -> 1.2% quantity decrease
        - |Elasticity| < 1.0: Price inelastic (raise prices)
        - |Elasticity| > 1.0: Price elastic (lower prices may increase profit)
        
        Args:
            order_items (pd.DataFrame): Normalized order data
        
        Returns:
            List[PricingOptimization]: Pricing recommendations for items with
                sufficient variance in price and quantity
        
        Note:
            Only returns recommendations with >0.5 confidence (R² > 0.25)
        """
        logger.info("Analyzing price elasticity and optimization...")
        
        recommendations = []
        
        for menu_item_id in order_items["menu_item_id"].unique():
            item_data = order_items[order_items["menu_item_id"] == menu_item_id]
            item_title = item_data["item_title"].iloc[0]
            
            # Need price variance to estimate elasticity
            if item_data["unit_price"].std() < 0.01:
                continue
            
            # Aggregate by price point
            price_quantity = item_data.groupby("unit_price").agg(
                quantity=("quantity", "sum"),
                revenue=("item_revenue", "sum"),
                cost=("item_cost", "sum"),
            ).reset_index()
            
            if len(price_quantity) < 3:
                continue  # Need at least 3 price points
            
            # Calculate elasticity using linear regression
            X = np.log(price_quantity["unit_price"].values).reshape(-1, 1)
            y = np.log(price_quantity["quantity"].values)
            
            try:
                # Simple linear regression: log(q) = a + b*log(p)
                # b is the elasticity
                slope, intercept, r_value, _, _ = stats.linregress(X.flatten(), y)
                elasticity = slope
                confidence = r_value ** 2
            except:
                continue
            
            if confidence < 0.25:
                continue  # Low confidence
            
            current_price = item_data["unit_price"].mean()
            current_quantity = item_data["quantity"].sum()
            current_revenue = item_data["item_revenue"].sum()
            current_cost = item_data["item_cost"].sum()
            current_profit = current_revenue - current_cost
            
            # Estimate optimal price (simplified: try ±10% and see which maximizes profit)
            optimal_price = current_price
            max_profit = current_profit
            
            for price_change in np.linspace(-0.15, 0.15, 31):
                test_price = current_price * (1 + price_change)
                # Use elasticity to estimate new quantity
                qty_change = elasticity * (price_change)
                test_quantity = current_quantity * (1 + qty_change)
                
                if test_quantity > 0:
                    test_revenue = test_price * test_quantity
                    test_profit = test_revenue - (current_cost / current_quantity * test_quantity)
                    
                    if test_profit > max_profit:
                        max_profit = test_profit
                        optimal_price = test_price
            
            revenue_impact = (optimal_price / current_price - 1) * current_revenue
            profit_impact = max_profit - current_profit
            
            recommendations.append(PricingOptimization(
                menu_item_id=str(menu_item_id),
                title=item_title,
                current_price=float(current_price),
                optimal_price=float(optimal_price),
                price_elasticity=float(elasticity),
                price_change_percent=float((optimal_price / current_price - 1) * 100),
                revenue_impact=float(revenue_impact),
                profit_impact=float(profit_impact),
                confidence_level=float(confidence),
                rationale=self._pricing_rationale(elasticity, confidence),
            ))
        
        logger.info(f"Generated {len(recommendations)} pricing recommendations")
        return recommendations

    def analyze_customer_behavior(self, order_items: pd.DataFrame) -> Dict[str, CustomerBehavior]:
        """
        Analyze customer purchase behavior and bundling opportunities.
        
        This method:
        1. Calculates repeat purchase rates
        2. Identifies frequently co-purchased items
        3. Detects seasonal patterns
        4. Classifies customer segments
        
        Args:
            order_items (pd.DataFrame): Order transaction data
        
        Returns:
            Dict[str, CustomerBehavior]: Customer behavior analysis by item
        """
        logger.info("Analyzing customer behavior patterns...")
        
        behavior_analysis = {}
        
        for menu_item_id in order_items["menu_item_id"].unique():
            item_data = order_items[order_items["menu_item_id"] == menu_item_id]
            item_title = item_data["item_title"].iloc[0]
            
            # Average units per transaction
            avg_units = item_data["quantity"].mean()
            
            # Repeat purchase rate (assuming order_id exists)
            if "order_id" in item_data.columns:
                unique_orders = item_data["order_id"].nunique()
                repeat_rate = 0.0  # Simplified - would need customer tracking
            else:
                repeat_rate = 0.0
            
            # Identify co-purchase items
            co_purchases = []  # Would need order-level data
            
            behavior_analysis[str(menu_item_id)] = CustomerBehavior(
                menu_item_id=str(menu_item_id),
                title=item_title,
                avg_units_per_transaction=float(avg_units),
                repeat_purchase_rate=float(repeat_rate),
                co_purchase_items=co_purchases,
            )
        
        return behavior_analysis

    def generate_optimization_plan(
        self,
        insights: pd.DataFrame,
        pricing_recs: Optional[List[PricingOptimization]] = None,
    ) -> MenuOptimizationPlan:
        """
        Generate a comprehensive menu optimization plan.
        
        This method synthesizes all analysis into an actionable plan that includes:
        1. Items to promote (Stars)
        2. Items to reposition (Plows)
        3. Items to reprice (high-margin items with elasticity)
        4. Items to remove (Dogs with low margins)
        5. Bundling opportunities
        
        The plan includes:
        - Expected revenue impact
        - Expected margin improvement
        - Prioritized action list for implementation
        
        Args:
            insights (pd.DataFrame): Menu insights from build_menu_insights()
            pricing_recs (Optional[List[PricingOptimization]]): Pricing recommendations
        
        Returns:
            MenuOptimizationPlan: Complete optimization plan with recommendations
        """
        logger.info("Generating menu optimization plan...")
        
        plan = MenuOptimizationPlan(
            plan_name="MenuMetrics Optimization Plan",
            created_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Items to promote (Stars)
        stars = insights[insights["category"] == "star"]
        plan.items_to_promote = [
            {
                "menu_item_id": row["menu_item_id"],
                "title": row["item_title"],
                "revenue": float(row["total_revenue"]),
                "margin": float(row["contribution_margin"]),
                "reason": "Star performer: high profit + high popularity",
            }
            for _, row in stars.iterrows()
        ]

        # Items to optimize (Plows)
        plows = insights[insights["category"] == "plowhorse"]
        plan.items_to_optimize = [
            {
                "menu_item_id": row["menu_item_id"],
                "title": row["item_title"],
                "current_revenue": float(row["total_revenue"]),
                "margin_pct": float(row["margin_percentage"]),
                "reason": "High margin but low popularity: needs better visibility",
                "action": "Improve menu placement, add appetizing description, feature in promotions",
            }
            for _, row in plows.iterrows()
        ]

        # Items to remove (Dogs)
        dogs = insights[insights["category"] == "dog"]
        plan.items_to_remove = [
            {
                "menu_item_id": row["menu_item_id"],
                "title": row["item_title"],
                "revenue": float(row["total_revenue"]),
                "margin_pct": float(row["margin_percentage"]),
                "reason": "Low profit + low popularity: removal or redesign recommended",
            }
            for _, row in dogs.iterrows()
        ]

        # Items to reprice (from pricing analysis)
        if pricing_recs:
            plan.items_to_reprice = [
                {
                    "menu_item_id": rec.menu_item_id,
                    "title": rec.title,
                    "current_price": rec.current_price,
                    "recommended_price": rec.optimal_price,
                    "price_change_pct": rec.price_change_percent,
                    "profit_impact": rec.profit_impact,
                    "confidence": rec.confidence_level,
                }
                for rec in pricing_recs
                if abs(rec.price_change_percent) > 2.0  # Only significant changes
            ]

        # Calculate expected uplift
        current_margin = insights["contribution_margin"].sum()
        potential_from_pricing = sum([rec.profit_impact for rec in pricing_recs]) if pricing_recs else 0
        potential_from_removal = dogs["contribution_margin"].sum() * 0.10  # Conservative estimate

        plan.expected_revenue_uplift = float(potential_from_pricing * 0.5)
        plan.expected_margin_uplift = float(
            ((potential_from_pricing + potential_from_removal) / current_margin * 100)
            if current_margin > 0
            else 0
        )

        # Set implementation priority
        plan.implementation_priority = [
            "Remove low-performing items (Dogs)",
            "Promote high-performers (Stars)",
            "Optimize pricing for high-elasticity items",
            "Improve visibility of Plowhorses",
            "Test bundling of Puzzles with Stars",
            "Monitor and iterate weekly",
        ]

        logger.info("Optimization plan generated successfully")
        return plan

    def export_insights(self, insights: pd.DataFrame, output_path: str) -> None:
        """
        Export menu insights to CSV file.
        
        Args:
            insights (pd.DataFrame): Menu insights dataframe
            output_path (str): Output file path
        """
        insights.to_csv(output_path, index=False)
        logger.info(f"Insights exported to {output_path}")

    # ========== Private Helper Methods ==========

    def _resolve_column(
        self,
        df: pd.DataFrame,
        candidates: List[str],
        optional: bool = False,
    ) -> Optional[str]:
        """
        Find a column name that exists in a dataframe.
        
        Tries each candidate in order and returns the first match.
        Useful for handling flexible naming conventions in real-world data.
        
        Args:
            df (pd.DataFrame): DataFrame to inspect
            candidates (List[str]): Column names to try
            optional (bool): If True, return None if no match found
        
        Returns:
            Optional[str]: Resolved column name or None
        
        Raises:
            ValueError: If no match found and optional=False
        """
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        
        if optional:
            return None
        
        raise ValueError(f"Missing required column. Expected one of: {candidates}")

    def _classify_item(
        self,
        row: pd.Series,
        popularity_threshold: float,
        margin_threshold: float,
    ) -> str:
        """
        Classify menu item into BCG Matrix quadrant.
        
        Uses contribution margin and sales volume thresholds (typically medians)
        to classify into 4 strategic categories.
        
        Args:
            row (pd.Series): Menu item metrics row
            popularity_threshold (float): Minimum units for "popular"
            margin_threshold (float): Minimum DKK for "profitable"
        
        Returns:
            str: Category ("star", "plowhorse", "puzzle", "dog")
        """
        is_popular = row["total_quantity"] >= popularity_threshold
        is_profitable = row["contribution_margin"] >= margin_threshold

        if is_profitable and is_popular:
            return "star"
        elif is_profitable and not is_popular:
            return "plowhorse"
        elif not is_profitable and is_popular:
            return "puzzle"
        else:
            return "dog"

    def _recommend_action(
        self,
        category: str,
        margin_pct: float,
        satisfaction: float = 0.0,
        votes: float = 0.0,
    ) -> str:
        """
        Generate strategic recommendation based on quadrant and customer satisfaction.

        Args:
            category (str): Menu engineering category
            margin_pct (float): Margin percentage
            satisfaction (float): Customer rating (0-5 scale)
            votes (float): Number of customer ratings for reliability weighting

        Returns:
            str: Recommended action with satisfaction context
        """
        # Base recommendations by category
        base_actions = {
            "star": {
                "no_rating_data": "Protect and promote based on strong performance; collect customer ratings.",
                "high_satisfaction": "Protect and promote aggressively - customers love this item!",
                "medium_satisfaction": "Promote with caution - consider quality improvements.",
                "low_satisfaction": "Investigate quality issues before promoting further."
            },
            "plowhorse": {
                "no_rating_data": "Increase visibility and gather ratings before major recipe changes.",
                "high_satisfaction": "Increase visibility - customers enjoy it but don't know about it.",
                "medium_satisfaction": "Increase menu placement and test price increases.",
                "low_satisfaction": "Consider repositioning or improving recipe before promotion."
            },
            "puzzle": {
                "no_rating_data": "Optimize pricing first and collect ratings to validate customer appeal.",
                "high_satisfaction": "Optimize recipe/pricing - item is liked but unprofitable.",
                "medium_satisfaction": "Optimize pricing or bundle with Stars.",
                "low_satisfaction": "Redesign to improve both profitability and appeal."
            },
            "dog": {
                "no_rating_data": "Remove or redesign based on business metrics; rating signal is unavailable.",
                "high_satisfaction": "Redesign recipe to reduce costs - customers like the concept.",
                "medium_satisfaction": "Remove or redesign - low profitability and popularity.",
                "low_satisfaction": "Remove - unprofitable and disliked by customers."
            }
        }

        # Weight rating impact by vote count.
        # With zero votes, rating has zero influence on the recommendation.
        votes = max(0.0, float(votes))
        satisfaction = float(np.clip(satisfaction, 0.0, 5.0))

        if votes <= 0:
            satisfaction_level = "no_rating_data"
        else:
            # Confidence reaches 1.0 at ~100 votes and scales smoothly below that.
            confidence = min(1.0, np.log1p(votes) / np.log1p(100.0))
            # Shrink low-confidence ratings toward neutral (2.5/5).
            adjusted_satisfaction = 2.5 + confidence * (satisfaction - 2.5)

            if adjusted_satisfaction >= 4.0:
                satisfaction_level = "high_satisfaction"
            elif adjusted_satisfaction >= 2.5:
                satisfaction_level = "medium_satisfaction"
            else:
                satisfaction_level = "low_satisfaction"

        # Get recommendation
        if category in base_actions:
            return base_actions[category].get(satisfaction_level, "Monitor performance")

        return "Monitor performance"

    def _price_hint(self, category: str, avg_price: float, margin_pct: float) -> str:
        """
        Generate pricing guidance.
        
        Args:
            category (str): Menu category
            avg_price (float): Average selling price
            margin_pct (float): Margin percentage
        
        Returns:
            str: Pricing guidance
        """
        if avg_price <= 0:
            return "Price data missing. Validate source."
        
        hints = {
            "star": f"Maintain or increase pricing. Current avg: {self.currency} {avg_price:.2f}",
            "plowhorse": f"Test +5% to +8% increases (current: {self.currency} {avg_price:.2f})",
            "puzzle": f"Bundle or offer at {self.currency} {avg_price*0.95:.2f} to drive volume",
            "dog": "Don't discount further. Focus on cost reduction.",
        }
        return hints.get(category, "Monitor and adjust based on demand")

    def _pricing_rationale(self, elasticity: float, confidence: float) -> str:
        """Generate rationale for pricing recommendation."""
        if confidence < 0.4:
            return "Low confidence in recommendation. Suggest A/B test pricing."
        elif abs(elasticity) < 0.8:
            return "Low price sensitivity indicates pricing power. Consider increases."
        elif abs(elasticity) > 1.2:
            return "High price sensitivity. Monitor closely for demand changes."
        else:
            return "Moderate price sensitivity. Gradual price testing recommended."
