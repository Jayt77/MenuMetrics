"""
File: streamlit_dashboard.py
Description: Interactive dashboard for menu engineering analysis and optimization.
Dependencies: streamlit, plotly, pandas
Author: MenuMetrics Intelligence Platform Team

This module provides a comprehensive Streamlit dashboard for stakeholders to:
1. View menu performance analytics (BCG Matrix visualization)
2. Analyze profitability and popularity metrics
3. Explore pricing optimization recommendations
4. Generate and download optimized menu plans
5. Compare current vs. optimized menu

The dashboard is the primary user interface for MenuMetrics Intelligence Platform.
Features are organized in tabs for easy navigation and stakeholder communication.

Usage:
    streamlit run src/streamlit_dashboard.py --logger.level=error
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from models.data_loader import DataLoader
from services.menu_engineering_service import MenuEngineeringService
from pages.landing_page import show_landing_page

# Configure logging (suppress verbose Streamlit logs)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="MenuMetrics Intelligence Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean Sparrow-style dashboard
st.markdown("""
<style>
    /* Force a single light theme to avoid mixed light/dark rendering */
    :root, [data-theme="dark"], [data-theme="light"] {
        color-scheme: light !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .stApp,
    html,
    body {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }

    /* Main content area - clean white background */
    .main {
        background-color: #FFFFFF;
        padding: 2rem;
    }

    /* Sidebar - light gray like Sparrow */
    [data-testid="stSidebar"] {
        background-color: #F7F8FA;
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1rem 1.5rem;
    }

    /* Sidebar navigation styling */
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        color: #1F2937;
        font-size: 14px;
        margin-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.25rem;
    }

    [data-testid="stSidebar"] .stRadio label {
        background-color: transparent;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        color: #6B7280;
        font-size: 14px;
        font-weight: 500;
        border: none;
        margin: 0;
        transition: all 0.2s;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #E5E7EB;
        color: #1F2937;
    }

    [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
    }

    /* KPI metric cards - white with border like Sparrow */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1F2937;
    }

    div[data-testid="stMetricLabel"] {
        color: #6B7280;
        font-size: 13px;
        font-weight: 500;
        text-transform: none;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 12px;
    }

    /* Status badges - cleaner style */
    .star-badge {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }

    .dog-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }

    .puzzle-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }

    .plow-badge {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }

    /* Loading overlay */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.95);
        z-index: 9999;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        backdrop-filter: blur(5px);
    }

    .loading-spinner {
        border: 4px solid #E5E7EB;
        border-top: 4px solid #1E88E5;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .loading-text {
        margin-top: 1rem;
        font-size: 16px;
        color: #374151;
        font-weight: 500;
    }

    /* Headers */
    h1, h2, h3 {
        color: #1F2937;
        font-weight: 700;
    }

    h1 {
        font-size: 32px;
        margin-bottom: 1.5rem;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 6px;
        font-weight: 500;
        transition: background-color 0.2s;
    }

    .stButton > button:hover {
        background-color: #1976D2;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ========== Session State Management ==========

def initialize_session_state():
    """Initialize Streamlit session state for data caching."""
    if "service" not in st.session_state:
        st.session_state.service = None
    if "insights" not in st.session_state:
        st.session_state.insights = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "order_items" not in st.session_state:
        st.session_state.order_items = None
    if "pricing_recs" not in st.session_state:
        st.session_state.pricing_recs = None
    if "optimization_plan" not in st.session_state:
        st.session_state.optimization_plan = None
    # Data upload session state
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    if "uploaded_menu" not in st.session_state:
        st.session_state.uploaded_menu = None
    if "all_uploaded_files" not in st.session_state:
        st.session_state.all_uploaded_files = None
    if "data_source" not in st.session_state:
        st.session_state.data_source = None  # 'uploaded', 'sample', or None
    if "data_profile" not in st.session_state:
        st.session_state.data_profile = None
    if "data_mappings" not in st.session_state:
        st.session_state.data_mappings = None
    # Pagination state
    if "show_more_counts" not in st.session_state:
        st.session_state.show_more_counts = {}
    # AI Assistant state
    if "ai_assistant" not in st.session_state:
        st.session_state.ai_assistant = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # Profitability assumptions
    if "profitability_margin" not in st.session_state:
        st.session_state.profitability_margin = 0.20  # 20% default
    # Language analysis cache
    if "language_analysis" not in st.session_state:
        st.session_state.language_analysis = None
    # Behavior analysis cache
    if "behavior_analysis" not in st.session_state:
        st.session_state.behavior_analysis = None


# ========== Data Loading Functions ==========

@st.cache_resource
def load_service():
    """Load MenuEngineeringService (cached)."""
    try:
        # Get absolute path to data directory
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(current_dir, "data")
        if not os.path.exists(data_dir):
            return None
        loader = DataLoader(data_dir)
        return MenuEngineeringService(loader)
    except Exception as e:
        st.error(f"Failed to initialize service: {e}")
        return None


@st.cache_resource
def load_ai_assistant():
    """Lazy-load AI assistant with Groq API key"""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            return None

        from ai.gemini_assistant import GeminiMenuAssistant
        return GeminiMenuAssistant(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to load AI assistant: {e}")
        return None


def get_show_more_count(key: str, increment: int = 25) -> int:
    """
    Get the current show count for a paginated list.

    Args:
        key: Unique identifier for the list
        increment: Number of items to show per page

    Returns:
        Current number of items to show
    """
    if key not in st.session_state.show_more_counts:
        st.session_state.show_more_counts[key] = increment
    return st.session_state.show_more_counts[key]


def show_more_button(key: str, total_items: int, increment: int = 25):
    """
    Display a 'Show More' button if there are more items to display.

    Args:
        key: Unique identifier for the list
        total_items: Total number of items available
        increment: Number of items to add when clicked
    """
    current_count = get_show_more_count(key, increment)

    if current_count < total_items:
        remaining = total_items - current_count
        button_text = f"Show More ({min(remaining, increment)} of {remaining} remaining)"

        if st.button(button_text, key=f"show_more_{key}", use_container_width=True):
            st.session_state.show_more_counts[key] = current_count + increment
            st.rerun()


def show_loading_overlay(message="Recalculating data, please wait..."):
    """Display full-screen loading overlay."""
    st.markdown(f"""
    <div class="loading-overlay">
        <div class="loading-spinner"></div>
        <div class="loading-text">{message}</div>
        <div style="margin-top: 10px; color: #666; font-size: 0.9em;">
            This may take a few moments with large datasets
        </div>
    </div>
    """, unsafe_allow_html=True)

def load_analysis_data(service, profitability_margin=0.20):
    """Load and analyze menu data with configurable profitability assumptions."""
    if service is None:
        st.error("Service not initialized. Check data directory.")
        return False

    # Show loading overlay
    placeholder = st.empty()
    with placeholder.container():
        show_loading_overlay(f"Loading and analyzing menu data (profit margin: {profitability_margin*100:.0f}%)...")

    try:
        order_items = service.load_menu_data(
            "Menu Engineering Part 2/dim_menu_items.csv",
            "Menu Engineering Part 1/fct_order_items.csv",
            profitability_margin=profitability_margin
        )

        insights = service.build_menu_insights(order_items)
        summary = service.build_summary(insights)

        st.session_state.order_items = order_items
        st.session_state.insights = insights
        st.session_state.summary = summary
        st.session_state.service = service

        # Clear loading overlay
        placeholder.empty()
        return True
    except FileNotFoundError as e:
        placeholder.empty()
        st.error(f"Data files not found: {e}")
        st.info("Please ensure Menu Engineering Part 2 is extracted to data/ directory")
        return False
    except Exception as e:
        placeholder.empty()
        st.error(f"Error loading data: {e}")
        return False


# ========== Visualization Functions ==========

def create_bcg_matrix_chart(insights, max_points=500):
    """
    Create optimized BCG Matrix visualization.

    Performance optimizations:
    - Samples top N items by revenue for large datasets
    - Uses WebGL rendering for smooth interaction
    - Removes text labels (only hover tooltips)
    - Optimized marker sizes

    Args:
        insights: DataFrame with menu insights
        max_points: Maximum number of points to display (default 500)
    """
    # Sample data if too large (keep top items by revenue)
    if len(insights) > max_points:
        sampled_insights = insights.nlargest(max_points, 'total_revenue')
        is_sampled = True
        sampled_count = len(insights) - max_points
    else:
        sampled_insights = insights
        is_sampled = False
        sampled_count = 0

    fig = go.Figure()

    for category, color, name in [
        ("star", "#FFD700", "Star"),
        ("plowhorse", "#45B7D1", "Plowhorse"),
        ("puzzle", "#4ECDC4", "Puzzle"),
        ("dog", "#FF6B6B", " Dog"),
    ]:
        cat_data = sampled_insights[sampled_insights["category"] == category]
        if len(cat_data) > 0:
            fig.add_trace(go.Scattergl(  # WebGL for performance
                x=cat_data["popularity_score"],
                y=cat_data["contribution_margin"],
                mode="markers",  # No text labels for performance
                name=name,
                text=cat_data["item_title"],
                customdata=cat_data[["total_revenue", "order_count"]],
                marker=dict(
                    size=np.clip(cat_data["total_revenue"] / 10000 + 5, 5, 20),  # Clipped size
                    color=color,
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="<b>%{text}</b><br>" +
                              "Popularity: %{x:.1f}%<br>" +
                              "Margin: DKK %{y:,.0f}<br>" +
                              "Revenue: DKK %{customdata[0]:,.0f}<br>" +
                              "Orders: %{customdata[1]:,.0f}<br>" +
                              "<extra></extra>",
            ))

    # Add threshold lines
    fig.add_hline(
        y=insights["contribution_margin"].median(),
        line_dash="dash",
        line_color="gray",
        annotation_text="Profit Threshold",
        annotation_position="right",
    )

    fig.add_vline(
        x=insights["popularity_score"].median(),
        line_dash="dash",
        line_color="gray",
        annotation_text="Popularity Threshold",
        annotation_position="top",
    )

    # Add sampling notice if applicable
    title_text = "<b>BCG Matrix: Menu Item Classification</b>"
    if is_sampled:
        title_text += f"<br><sub>Showing top {max_points} items by revenue ({sampled_count:,} items hidden for performance)</sub>"

    fig.update_layout(
        title=title_text,
        xaxis_title="Popularity Score (%)",
        yaxis_title="Contribution Margin (DKK)",
        height=500,
        hovermode="closest",
        template="plotly_white",
        # Performance optimizations
        dragmode='pan',  # Faster than zoom
    )

    return fig


def create_category_distribution_chart(insights):
    """Create pie chart of items by category."""
    category_counts = insights["category"].value_counts()
    colors = {
        "star": "#FFD700",
        "plowhorse": "#45B7D1",
        "puzzle": "#4ECDC4",
        "dog": "#FF6B6B",
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=[cat.upper() for cat in category_counts.index],
        values=category_counts.values,
        marker=dict(colors=[colors.get(cat, "#999") for cat in category_counts.index]),
        hovertemplate="<b>%{label}</b><br>Items: %{value}<extra></extra>",
    )])
    
    fig.update_layout(
        title="<b>Menu Distribution by Category</b>",
        height=400,
    )
    
    return fig


def create_revenue_by_category_chart(insights):
    """Create bar chart of revenue and margin by category."""
    category_data = insights.groupby("category").agg({
        "total_revenue": "sum",
        "contribution_margin": "sum",
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=category_data["category"].str.upper(),
        y=category_data["total_revenue"],
        name="Revenue",
        marker_color="#3498DB",
    ))
    
    fig.add_trace(go.Bar(
        x=category_data["category"].str.upper(),
        y=category_data["contribution_margin"],
        name="Margin",
        marker_color="#2ECC71",
    ))
    
    fig.update_layout(
        title="<b>Revenue & Margin by Category</b>",
        xaxis_title="Category",
        yaxis_title="DKK",
        barmode="group",
        height=400,
    )
    
    return fig


def create_top_items_table(insights, top_n=10):
    """Create table of top items by revenue."""
    top_items = insights.nlargest(top_n, "total_revenue")[
        ["item_title", "category", "total_quantity", "total_revenue", "margin_percentage"]
    ].copy()
    
    top_items = top_items.rename(columns={
        "item_title": "Item",
        "category": "Category",
        "total_quantity": "Units Sold",
        "total_revenue": "Revenue (DKK)",
        "margin_percentage": "Margin %",
    })
    
    return top_items


def create_pricing_chart(pricing_recs):
    """Create chart comparing current vs. recommended prices."""
    if not pricing_recs:
        return None
    
    recs_df = pd.DataFrame([
        {
            "Title": rec.title[:20],
            "Current": rec.current_price,
            "Recommended": rec.optimal_price,
            "Impact": rec.profit_impact,
        }
        for rec in pricing_recs
        if abs(rec.price_change_percent) > 2.0  # Only show significant changes
    ])
    
    if recs_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=recs_df["Title"],
        y=recs_df["Current"],
        name="Current Price",
        marker_color="#3498DB",
    ))
    
    fig.add_trace(go.Bar(
        x=recs_df["Title"],
        y=recs_df["Recommended"],
        name="Recommended Price",
        marker_color="#2ECC71",
    ))
    
    fig.update_layout(
        title="<b>Price Optimization Recommendations</b>",
        xaxis_title="Item",
        yaxis_title="Price (DKK)",
        barmode="group",
        height=400,
        xaxis_tickangle=-45,
    )
    
    return fig


# ========== Dashboard Pages ==========

def page_overview():
    """Dashboard overview tab."""
    st.header("[Data] Menu Performance Overview")
    
    if st.session_state.insights is None:
        st.warning("Data is loading, please wait...")
        return
    
    insights = st.session_state.insights
    summary = st.session_state.summary
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Items",
            f"{summary['total_items']:.0f}",
            help="Number of menu items"
        )

    with col2:
        st.metric(
            "Total Revenue",
            f"DKK {summary['total_revenue']:,.0f}",
            help="Total historical revenue"
        )

    with col3:
        st.metric(
            "Total Margin",
            f"DKK {summary['total_margin']:,.0f}",
            help="Total contribution margin"
        )

    with col4:
        st.metric(
            "Avg Margin %",
            f"{summary['average_margin_pct']:.1f}%",
            help="Average margin percentage"
        )

    with col5:
        # Calculate average customer rating if available
        if "rating_normalized" in insights.columns:
            avg_rating = insights[insights["rating_normalized"] > 0]["rating_normalized"].mean()
            total_votes = insights["votes"].sum()
            st.metric(
                "Avg Customer Rating",
                f"{avg_rating:.2f}/5",
                help=f"Average customer rating ({total_votes:,.0f} total votes)"
            )
        else:
            st.metric(
                "Avg Customer Rating",
                "N/A",
                help="Customer ratings not available"
            )

    # Display profitability assumption
    st.caption(f"Profitability Assumption: {st.session_state.profitability_margin * 100:.0f}% (adjustable in sidebar)")

    st.divider()

    # BCG Matrix with view toggle
    view_mode = st.radio(
        "Matrix View",
        ["Detailed Scatter Plot", "Simplified Quadrant Summary"],
        horizontal=True,
        help="Choose visualization style. Simplified view is faster for large datasets."
    )

    if view_mode == "Detailed Scatter Plot":
        col1, col2 = st.columns([2, 1])

        with col1:
            # Allow user to adjust max points
            if len(insights) > 500:
                max_points = st.slider(
                    "Max points to display",
                    min_value=100,
                    max_value=min(2000, len(insights)),
                    value=500,
                    step=100,
                    help="Reduce for better performance with large datasets"
                )
            else:
                max_points = len(insights)

            st.plotly_chart(create_bcg_matrix_chart(insights, max_points), use_container_width=True)

        with col2:
            st.plotly_chart(create_category_distribution_chart(insights), use_container_width=True)

    else:
        # Simplified Quadrant Summary View
        st.subheader("Quadrant Summary")

        # Create 2x2 grid for quadrants
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        quadrants = [
            ("star", " STARS", "High Profit + High Popularity", row1_col1, "#FFD700"),
            ("plowhorse", " PLOWHORSES", "High Profit + Low Popularity", row1_col2, "#45B7D1"),
            ("puzzle", " PUZZLES", "Low Profit + High Popularity", row2_col1, "#4ECDC4"),
            ("dog", " DOGS", "Low Profit + Low Popularity", row2_col2, "#FF6B6B"),
        ]

        for category, title, description, col, color in quadrants:
            with col:
                cat_data = insights[insights["category"] == category]

                if len(cat_data) > 0:
                    total_revenue = cat_data["total_revenue"].sum()
                    total_margin = cat_data["contribution_margin"].sum()
                    avg_margin_pct = cat_data["margin_percentage"].mean()
                    item_count = len(cat_data)

                    st.markdown(f"""
                    <div style="background-color: {color}22; border-left: 4px solid {color}; padding: 15px; border-radius: 5px; height: 220px;">
                        <h3 style="color: {color}; margin-top: 0;">{title}</h3>
                        <p style="color: #666; font-size: 0.9em; margin-bottom: 15px;">{description}</p>
                        <div style="margin-top: 10px;">
                            <strong>{item_count:,}</strong> items<br>
                            <strong>DKK {total_revenue:,.0f}</strong> revenue<br>
                            <strong>DKK {total_margin:,.0f}</strong> margin<br>
                            <strong>{avg_margin_pct:.1f}%</strong> avg margin
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: {color}22; border-left: 4px solid {color}; padding: 15px; border-radius: 5px; height: 220px;">
                        <h3 style="color: {color}; margin-top: 0;">{title}</h3>
                        <p style="color: #666; font-size: 0.9em;">No items in this category</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Revenue analysis
    st.plotly_chart(create_revenue_by_category_chart(insights), use_container_width=True)
    
    # Top items table
    st.subheader(" Top 10 Items by Revenue")
    top_items = create_top_items_table(insights, 10)
    st.dataframe(
        top_items.style.format({
            "Units Sold": "{:,.0f}",
            "Revenue (DKK)": "{:,.0f}",
            "Margin %": "{:.1f}%"
        }),
        use_container_width=True,
        hide_index=True,
    )


def page_category_analysis():
    """Deep dive into each BCG category."""
    st.header("[Target] Category Analysis")
    
    if st.session_state.insights is None:
        st.warning("Data is loading, please wait...")
        return
    
    insights = st.session_state.insights
    
    # Select category
    selected_category = st.radio(
        "Select Category",
        ["star", "plowhorse", "puzzle", "dog"],
        format_func=lambda x: {
            "star": "Stars (High profit + High popularity)",
            "plowhorse": "Plowhorses (High profit + Low popularity)",
            "puzzle": "Puzzles (Low profit + High popularity)",
            "dog": " Dogs (Low profit + Low popularity)",
        }.get(x, x),
        horizontal=True,
    )
    
    cat_items = insights[insights["category"] == selected_category]
    
    if len(cat_items) == 0:
        st.info(f"No items in {selected_category} category")
        return
    
    # Category summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Items in Category", len(cat_items))
    with col2:
        st.metric("Total Revenue", f"DKK {cat_items['total_revenue'].sum():,.0f}")
    with col3:
        st.metric("Total Margin", f"DKK {cat_items['contribution_margin'].sum():,.0f}")
    with col4:
        st.metric("Avg Margin %", f"{cat_items['margin_percentage'].mean():.1f}%")
    
    st.divider()
    
    # Category description and recommendations
    category_info = {
        "star": {
            "description": "These are your best performers - high profits and high popularity.",
            "action": "Protect availability, promote aggressively, consider premium variants.",
            "color": "",
        },
        "plowhorse": {
            "description": "High profit items with low sales volume.",
            "action": "Increase visibility through menu placement, description enhancement, and targeted promotions.",
            "color": "",
        },
        "puzzle": {
            "description": "Popular items with low profit margins.",
            "action": "Optimize pricing, test bundles with Stars, enhance descriptions.",
            "color": "",
        },
        "dog": {
            "description": "Low profits and low popularity.",
            "action": "Remove or redesign. If margins < 5%, removal is recommended.",
            "color": "",
        },
    }
    
    info = category_info.get(selected_category, {})
    st.info(f"**{info['description']}**\n\n**Recommended Action:** {info['action']}")
    
    st.divider()
    
    # Items in category
    st.subheader(f"Items in {selected_category.upper()} Category")

    # Build display columns (include customer rating if available)
    display_cols = ["item_title", "total_quantity", "total_revenue", "contribution_margin", "margin_percentage"]

    # Add customer rating columns if available
    if "rating_normalized" in cat_items.columns and "votes" in cat_items.columns:
        display_cols.extend(["rating_normalized", "votes"])

    display_cols.append("suggested_action")

    category_table = cat_items[display_cols].copy()

    # Rename columns
    rename_dict = {
        "item_title": "Item",
        "total_quantity": "Units Sold",
        "total_revenue": "Revenue (DKK)",
        "contribution_margin": "Margin (DKK)",
        "margin_percentage": "Margin %",
        "suggested_action": "Recommendation",
    }

    if "rating_normalized" in category_table.columns:
        rename_dict["rating_normalized"] = "Customer Rating"
        rename_dict["votes"] = "Votes"

    category_table = category_table.rename(columns=rename_dict)

    # Show More functionality
    pagination_key = f"category_{selected_category}"
    show_count = get_show_more_count(pagination_key)
    total_items = len(category_table)

    # Display limited items with formatting
    format_dict = {
        "Units Sold": "{:,.0f}",
        "Revenue (DKK)": "{:,.0f}",
        "Margin (DKK)": "{:,.0f}",
        "Margin %": "{:.1f}%"
    }

    # Add customer rating formatting if present
    if "Customer Rating" in category_table.columns:
        format_dict["Customer Rating"] = "{:.2f}/5"
        format_dict["Votes"] = "{:,.0f}"

    st.dataframe(
        category_table.head(show_count).style.format(format_dict),
        use_container_width=True,
        hide_index=True,
    )

    # Show More button
    show_more_button(pagination_key, total_items)


def page_pricing_analysis():
    """Pricing optimization tab."""
    st.header(" Pricing Optimization")
    
    if st.session_state.insights is None:
        st.warning("Data is loading, please wait...")
        return
    
    # Generate pricing recommendations if not already done
    if st.session_state.pricing_recs is None:
        with st.spinner("Analyzing price elasticity..."):
            st.session_state.pricing_recs = st.session_state.service.analyze_pricing_optimization(
                st.session_state.order_items
            )
    
    pricing_recs = st.session_state.pricing_recs
    
    if not pricing_recs:
        st.info("Insufficient price variation in data for elasticity analysis")
        return
    
    st.subheader(f"Found {len(pricing_recs)} items with pricing opportunities")
    
    # Filter by confidence level
    confidence_threshold = st.slider(
        "Minimum Confidence Level",
        0.0, 1.0, 0.5,
        help="Only show recommendations with this confidence or higher"
    )
    
    filtered_recs = [r for r in pricing_recs if r.confidence_level >= confidence_threshold]
    
    if not filtered_recs:
        st.info(f"No recommendations found with >= {confidence_threshold*100:.0f}% confidence")
        return
    
    # Chart
    chart = create_pricing_chart(filtered_recs)
    if chart:
        st.plotly_chart(chart, use_container_width=True)
    
    st.divider()
    
    # Recommendations table
    st.subheader("Detailed Recommendations")

    recs_data = []
    for rec in filtered_recs:
        if abs(rec.price_change_percent) > 1.0:  # Only significant changes
            recs_data.append({
                "Item": rec.title[:25],
                "Current Price": f"DKK {rec.current_price:.2f}",
                "Recommended": f"DKK {rec.optimal_price:.2f}",
                "Change": f"{rec.price_change_percent:+.1f}%",
                "Profit Impact": f"DKK {rec.profit_impact:+,.0f}",
                "Confidence": f"{rec.confidence_level*100:.0f}%",
                "Elasticity": f"{rec.price_elasticity:.2f}",
            })

    if recs_data:
        recs_df = pd.DataFrame(recs_data)
        st.dataframe(recs_df, use_container_width=True, hide_index=True)
    
    st.info("""
    **How to interpret elasticity:**
    - **Elasticity = -0.8**: 1% price increase -> 0.8% quantity decrease (inelastic, good for raising prices)
    - **Elasticity = -1.5**: 1% price increase -> 1.5% quantity decrease (elastic, price sensitive)
    - **Confidence**: Higher = more reliable recommendation (0-100%)
    """)


def page_optimization_plan():
    """Menu optimization plan tab."""
    st.header(" Menu Optimization Plan")

    if st.session_state.insights is None:
        st.warning("Data is loading, please wait...")
        return

    # Generate plan if not already done
    if st.session_state.optimization_plan is None:
        with st.spinner("Generating optimization plan..."):
            # Get pricing recs if not done
            if st.session_state.pricing_recs is None:
                st.session_state.pricing_recs = st.session_state.service.analyze_pricing_optimization(
                    st.session_state.order_items
                )

            st.session_state.optimization_plan = st.session_state.service.generate_optimization_plan(
                st.session_state.insights,
                st.session_state.pricing_recs,
            )

    plan = st.session_state.optimization_plan

    # Expected impact
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Expected Revenue Uplift",
            f"DKK {plan.expected_revenue_uplift:,.0f}",
            help="Potential additional revenue from optimization"
        )
    with col2:
        st.metric(
            "Expected Margin Improvement",
            f"{plan.expected_margin_uplift:.1f}%",
            help="Potential margin improvement percentage"
        )

    st.divider()

    # Implementation priority
    st.subheader(" Implementation Priority")
    for i, action in enumerate(plan.implementation_priority, 1):
        st.write(f"{i}. {action}")

    st.divider()

    # Items to promote
    if plan.items_to_promote:
        st.subheader("Star Items to Promote (Stars)")
        promote_df = pd.DataFrame(plan.items_to_promote)[["title", "revenue", "margin"]]
        promote_df = promote_df.rename(columns={"title": "Item", "revenue": "Revenue (DKK)", "margin": "Margin (DKK)"})
        st.dataframe(
            promote_df.style.format({"Revenue (DKK)": "{:,.0f}", "Margin (DKK)": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # Items to optimize
    if plan.items_to_optimize:
        st.subheader("Plowhorse Items to Reposition (Plows)")
        st.write("High-margin items that need better visibility:")
        for item in plan.items_to_optimize:
            st.write(f"- **{item['title']}** ({item['margin_pct']:.1f}% margin): {item['action']}")

    # Items to remove
    if plan.items_to_remove:
        st.subheader(" Items to Remove (Dogs)")
        remove_df = pd.DataFrame(plan.items_to_remove)[["title", "revenue", "margin_pct"]]
        remove_df = remove_df.rename(columns={"title": "Item", "revenue": "Revenue (DKK)", "margin_pct": "Margin %"})
        st.dataframe(
            remove_df.style.format({"Revenue (DKK)": "{:,.0f}", "Margin %": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True,
        )

    # Items to reprice
    if plan.items_to_reprice:
        st.subheader(" Pricing Recommendations")
        reprice_df = pd.DataFrame(plan.items_to_reprice)[
            ["title", "current_price", "recommended_price", "price_change_pct", "profit_impact"]
        ]
        reprice_df = reprice_df.rename(columns={
            "title": "Item",
            "current_price": "Current (DKK)",
            "recommended_price": "Recommended (DKK)",
            "price_change_pct": "Change %",
            "profit_impact": "Profit Impact (DKK)",
        })
        st.dataframe(
            reprice_df.style.format({
                "Current (DKK)": "{:.2f}",
                "Recommended (DKK)": "{:.2f}",
                "Change %": "{:+.1f}%",
                "Profit Impact (DKK)": "{:+,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # Download plan
    st.subheader("[Download] Download Plan")

    plan_text = f"""
MENU OPTIMIZATION PLAN
Generated: {plan.created_date}

=== EXECUTIVE SUMMARY ===
Expected Revenue Uplift: DKK {plan.expected_revenue_uplift:,.0f}
Expected Margin Improvement: {plan.expected_margin_uplift:.1f}%

=== ITEMS TO PROMOTE ({len(plan.items_to_promote)} items) ===
"""
    for item in plan.items_to_promote:
        plan_text += f"- {item['title']}: DKK {item['revenue']:,.0f} revenue, DKK {item['margin']:,.0f} margin\n"

    plan_text += f"\n=== ITEMS TO REMOVE ({len(plan.items_to_remove)} items) ===\n"
    for item in plan.items_to_remove:
        plan_text += f"- {item['title']}: {item['margin_pct']:.1f}% margin\n"

    plan_text += f"\n=== PRICING RECOMMENDATIONS ({len(plan.items_to_reprice)} items) ===\n"
    for item in plan.items_to_reprice:
        plan_text += f"- {item['title']}: DKK {item['current_price']:.2f} -> DKK {item['recommended_price']:.2f} ({item['price_change_pct']:+.1f}%)\n"

    st.download_button(
        "[HTML] Download Plan as Text",
        plan_text,
        "menu_optimization_plan.txt",
        "text/plain",
    )


def page_menu_language_analysis():
    """Menu language and keyword performance analysis."""
    st.header(" Menu Language Analysis")

    if st.session_state.insights is None:
        st.warning("Data is loading, please wait...")
        return

    insights = st.session_state.insights

    # Generate language analysis if not cached
    if "language_analysis" not in st.session_state or st.session_state.language_analysis is None:
        placeholder = st.empty()
        with placeholder.container():
            show_loading_overlay("Analyzing menu language patterns...")

        st.session_state.language_analysis = st.session_state.service.analyze_menu_language_correlation(insights)
        placeholder.empty()

    lang = st.session_state.language_analysis

    if not lang:
        st.error("Insufficient data for language analysis")
        return

    # KPIs row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Power Words Found", len(lang.get('power_words', [])))
    with col2:
        st.metric("Toxic Words Found", len(lang.get('toxic_words', [])))
    with col3:
        st.metric("Rename Suggestions", len(lang.get('rename_suggestions', [])))

    st.divider()

    # Power Words Section
    st.subheader(" Power Words (Revenue Boosters)")
    st.write("These terms correlate with significantly higher revenue when used in item names.")

    power_words = lang.get('power_words', [])
    if power_words:
        power_df = pd.DataFrame(power_words)
        power_df_display = power_df[['term', 'frequency', 'revenue_lift', 'revenue_lift_pct', 'p_value']].copy()
        power_df_display = power_df_display.rename(columns={
            'term': 'Term',
            'frequency': 'Items',
            'revenue_lift': 'Revenue Lift (DKK)',
            'revenue_lift_pct': 'Lift %',
            'p_value': 'P-Value'
        })

        st.dataframe(
            power_df_display.style.format({
                'Revenue Lift (DKK)': '{:,.0f}',
                'Lift %': '{:.1f}%',
                'P-Value': '{:.4f}'
            }).background_gradient(subset=['Lift %'], cmap='Greens'),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No statistically significant power words found")

    st.divider()

    # Toxic Words Section
    st.subheader(" Toxic Words (Revenue Drags)")
    st.write("These terms correlate with lower revenue - avoid using them in item names.")

    toxic_words = lang.get('toxic_words', [])
    if toxic_words:
        toxic_df = pd.DataFrame(toxic_words)
        toxic_df_display = toxic_df[['term', 'frequency', 'revenue_lift', 'revenue_lift_pct']].copy()
        toxic_df_display = toxic_df_display.rename(columns={
            'term': 'Term',
            'frequency': 'Items',
            'revenue_lift': 'Revenue Impact (DKK)',
            'revenue_lift_pct': 'Impact %'
        })

        st.dataframe(
            toxic_df_display.style.format({
                'Revenue Impact (DKK)': '{:,.0f}',
                'Impact %': '{:.1f}%'
            }).background_gradient(subset=['Impact %'], cmap='Reds_r'),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No toxic words detected")

    st.divider()

    # Category-Specific Terms
    st.subheader(" Category-Specific Language Patterns")

    category_terms = lang.get('category_terms', {})
    col1, col2 = st.columns(2)

    with col1:
        st.write("**STAR Items Common Terms:**")
        star_terms = category_terms.get('star', [])
        if star_terms:
            st.write(", ".join([f"**{term}**" for term in star_terms]))
        else:
            st.write("N/A")

        st.write("**PUZZLE Items Common Terms:**")
        puzzle_terms = category_terms.get('puzzle', [])
        if puzzle_terms:
            st.write(", ".join([f"**{term}**" for term in puzzle_terms]))
        else:
            st.write("N/A")

    with col2:
        st.write("**PLOWHORSE Items Common Terms:**")
        plow_terms = category_terms.get('plowhorse', [])
        if plow_terms:
            st.write(", ".join([f"**{term}**" for term in plow_terms]))
        else:
            st.write("N/A")

        st.write("**DOG Items Common Terms:**")
        dog_terms = category_terms.get('dog', [])
        if dog_terms:
            st.write(", ".join([f"**{term}**" for term in dog_terms]))
        else:
            st.write("N/A")

    st.divider()

    # Rename Suggestions
    st.subheader(" Suggested Item Renames")
    st.write("Based on language analysis, consider these renames to improve appeal:")

    rename_suggestions = lang.get('rename_suggestions', [])
    if rename_suggestions:
        for suggestion in rename_suggestions:
            with st.expander(f"{suggestion['current_name']} → {suggestion['suggested_name']}"):
                st.write(f"**Reason:** {suggestion['reason']}")
    else:
        st.info("No rename suggestions at this time")


def page_customer_insights():
    """Customer behavior pattern analysis and cross-sell opportunities."""
    st.header(" Customer Insights & Behavior Patterns")

    if st.session_state.insights is None or st.session_state.order_items is None:
        st.warning("Data is loading, please wait...")
        return

    # Generate behavior analysis if not cached
    if "behavior_analysis" not in st.session_state or st.session_state.behavior_analysis is None:
        placeholder = st.empty()
        with placeholder.container():
            show_loading_overlay("Analyzing customer behavior patterns...")

        order_items = st.session_state.order_items

        # Try to load timeline data
        timeline_data = None
        try:
            import os
            timeline_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "order_timeline_data.csv")
            if os.path.exists(timeline_path):
                timeline_data = pd.read_csv(timeline_path)
        except Exception as e:
            pass  # Timeline data optional

        st.session_state.behavior_analysis = st.session_state.service.analyze_purchase_patterns(
            order_items,
            timeline_data
        )
        placeholder.empty()

    behavior = st.session_state.behavior_analysis

    # Sub-tabs
    tab1, tab2, tab3 = st.tabs([
        " Co-Purchase Patterns",
        " Temporal Patterns",
        " Cross-Sell Opportunities"
    ])

    with tab1:
        st.subheader("Frequently Co-Purchased Items")
        st.write("Items that customers often buy together (market basket analysis)")

        copurchase_pairs = behavior.get('copurchase_pairs', [])
        if copurchase_pairs:
            # KPIs
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Co-Purchase Pairs Found", len(copurchase_pairs))
            with col2:
                avg_lift = np.mean([pair['lift'] for pair in copurchase_pairs])
                st.metric("Avg Lift", f"{avg_lift:.2f}x")
            with col3:
                avg_conf = np.mean([pair['confidence'] for pair in copurchase_pairs])
                st.metric("Avg Confidence", f"{avg_conf*100:.0f}%")

            st.divider()

            # Table of top pairs
            copurchase_df = pd.DataFrame(copurchase_pairs)
            copurchase_display = copurchase_df[['item_a', 'item_b', 'support', 'confidence', 'lift', 'count']].copy()
            copurchase_display = copurchase_display.rename(columns={
                'item_a': 'Item A',
                'item_b': 'Item B',
                'support': 'Support',
                'confidence': 'Confidence',
                'lift': 'Lift',
                'count': 'Orders'
            })

            st.dataframe(
                copurchase_display.style.format({
                    'Support': '{:.1%}',
                    'Confidence': '{:.1%}',
                    'Lift': '{:.2f}x',
                    'Orders': '{:,.0f}'
                }).background_gradient(subset=['Lift'], cmap='Blues'),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "**Lift** indicates how much more likely items are bought together vs random chance. "
                "Lift > 1.5 indicates a strong association."
            )
        else:
            st.info("No co-purchase patterns detected. Ensure order_id is available in data.")

    with tab2:
        st.subheader("Time-Based Ordering Patterns")

        hourly_patterns = behavior.get('hourly_patterns', {})
        peak_hours = behavior.get('peak_hours', [])

        if hourly_patterns or peak_hours:
            # Peak hours
            if peak_hours:
                st.write(f"**Peak Ordering Hours:** {', '.join([f'{h}:00' for h in peak_hours])}")
                st.divider()

            # Hourly breakdown
            if hourly_patterns:
                st.write("**Popular Items by Hour:**")

                # Create a heatmap-like display
                hours_to_show = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

                for hour in hours_to_show:
                    if hour in hourly_patterns:
                        items = hourly_patterns[hour]
                        if items:
                            with st.expander(f"{hour}:00 - {hour+1}:00", expanded=(hour in peak_hours[:3])):
                                for item_data in items[:5]:
                                    item_name = item_data.get('item_title', 'Unknown')
                                    count = item_data.get('count', 0)
                                    st.write(f"- **{item_name}**: {count:,} orders")

            # Day-of-week patterns
            st.divider()
            day_patterns = behavior.get('day_patterns', {})
            if day_patterns:
                st.write("**Weekday vs Weekend Favorites:**")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Weekday Favorites:**")
                    weekday_favs = day_patterns.get('weekday_favorites', {})
                    for item, count in list(weekday_favs.items())[:5]:
                        st.write(f"- {item}: {count:,} orders")

                with col2:
                    st.write("**Weekend Favorites:**")
                    weekend_favs = day_patterns.get('weekend_favorites', {})
                    for item, count in list(weekend_favs.items())[:5]:
                        st.write(f"- {item}: {count:,} orders")
        else:
            st.info("Temporal data not available. Add timestamp data to enable time-based analysis.")

    with tab3:
        st.subheader("Cross-Sell Opportunities")
        st.write("Recommended item pairings to increase order value")

        cross_sell = behavior.get('cross_sell_opportunities', [])
        if cross_sell:
            for i, opp in enumerate(cross_sell[:10], 1):
                with st.expander(f"#{i}: {opp['anchor_item']} → {opp['recommended_item']}"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.write(f"**Why:** {opp['rationale']}")

                    with col2:
                        st.metric("Attachment Rate", f"{opp['expected_rate']*100:.0f}%")
                        st.metric("Lift", f"{opp['lift']:.2f}x")
        else:
            st.info("No cross-sell opportunities detected")


def page_menu_generator():
    """
    Generate optimized menu tab.

    This page allows users to:
    1. Configure menu settings (items to include, layout preferences)
    2. Preview the optimized menu design
    3. Download the menu in multiple formats (HTML, PDF, TXT)

    The generator automatically includes Stars, selectively includes Plows,
    and excludes Dogs based on the BCG Matrix analysis.
    """
    st.header(" Generate Optimized Menu")

    if st.session_state.insights is None:
        st.warning("Data is loading, please wait...")
        return

    insights = st.session_state.insights

    st.write("""
    This tool generates an optimized menu based on your analysis. The menu automatically:
    - **Highlights Stars** (high profit + high popularity)
    - **Includes Plows** (high profit items that need visibility)
    - **Excludes Dogs** (low profit + low popularity items)
    - **Applies optimal pricing** from the pricing analysis
    """)

    st.divider()

    # Menu configuration
    st.subheader(" Menu Configuration")

    col1, col2 = st.columns(2)

    with col1:
        include_stars = st.checkbox("Include Stars", value=True, disabled=True, help="Stars are always included")
        include_plows = st.checkbox("Include Plows (High-margin items)", value=True)
        include_puzzles = st.checkbox("Include Puzzles (Popular items)", value=False)

    with col2:
        min_margin = st.slider(
            "Minimum Margin %",
            0.0, 100.0, 20.0,
            help="Only include items with at least this margin percentage"
        )
        max_items = st.number_input(
            "Maximum Items on Menu",
            min_value=5, max_value=100, value=25,
            help="Limit total menu items for simplicity"
        )

    # Filter items based on configuration
    filtered_items = insights[
        (insights["category"] == "star") |
        ((insights["category"] == "plowhorse") & include_plows) |
        ((insights["category"] == "puzzle") & include_puzzles)
    ].copy()

    # Apply margin filter
    filtered_items = filtered_items[filtered_items["margin_percentage"] >= min_margin]

    # Sort by strategic importance (Stars first, then by margin)
    category_order = {"star": 1, "plowhorse": 2, "puzzle": 3, "dog": 4}
    filtered_items["sort_order"] = filtered_items["category"].map(category_order)
    filtered_items = filtered_items.sort_values(["sort_order", "contribution_margin"], ascending=[True, False])

    # Limit to max items
    filtered_items = filtered_items.head(max_items)

    st.divider()

    # Preview metrics
    st.subheader("[Data] Optimized Menu Preview")

    if len(filtered_items) == 0:
        st.warning("No items match the current filters. Try lowering the minimum margin % or including more categories.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Items", len(filtered_items))
        with col2:
            # Calculate weighted average margin (by revenue)
            total_rev = filtered_items['total_revenue'].sum()
            if total_rev > 0:
                weighted_margin = (filtered_items['contribution_margin'].sum() / total_rev) * 100
                st.metric("Avg Margin %", f"{weighted_margin:.1f}%")
            else:
                st.metric("Avg Margin %", "N/A")
        with col3:
            st.metric("Total Revenue", f"DKK {filtered_items['total_revenue'].sum():,.0f}")
        with col4:
            st.metric("Total Profit", f"DKK {filtered_items['contribution_margin'].sum():,.0f}")

    st.divider()

    # Category breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Items by Category")
        category_counts = filtered_items["category"].value_counts()
        for cat in ["star", "plowhorse", "puzzle"]:
            if cat in category_counts.index:
                emoji = {"star": "Star", "plowhorse": "Plowhorse", "puzzle": "Puzzle"}.get(cat, "")
                st.write(f"{emoji} **{cat.title()}**: {category_counts[cat]} items")

    with col2:
        st.subheader("Menu Health Score")
        # Calculate menu health score based on:
        # - % of Stars (positive)
        # - Average margin (positive)
        # - Simplicity (fewer items = higher score)

        if len(filtered_items) > 0:
            star_pct = (filtered_items["category"] == "star").sum() / len(filtered_items) * 100
            avg_margin = filtered_items["margin_percentage"].mean()
            simplicity_score = max(0, 100 - len(filtered_items))

            health_score = (star_pct * 0.4 + avg_margin * 0.4 + simplicity_score * 0.2)

            # Ensure health_score is valid and within bounds
            if pd.isna(health_score) or not np.isfinite(health_score):
                health_score = 0
            else:
                health_score = max(0, min(100, health_score))
        else:
            health_score = 0

        st.metric("Overall Health Score", f"{health_score:.0f}/100")
        st.progress(health_score / 100)

        if health_score >= 80:
            st.success("Yes Excellent menu composition!")
        elif health_score >= 60:
            st.info("Yes Good menu with room for improvement")
        else:
            st.warning("[Warning] Consider adding more high-margin items")

    st.divider()

    # Menu items table
    st.subheader(" Menu Items")

    display_menu = filtered_items[["item_title", "category", "avg_price", "margin_percentage", "suggested_action"]].copy()
    display_menu = display_menu.rename(columns={
        "item_title": "Item",
        "category": "Category",
        "avg_price": "Price (DKK)",
        "margin_percentage": "Margin %",
        "suggested_action": "Notes",
    })

    # Show More functionality
    menu_key = "menu_generator_items"
    menu_show_count = get_show_more_count(menu_key)
    total_menu_items = len(display_menu)

    st.dataframe(
        display_menu.head(menu_show_count).style.format({
            "Price (DKK)": "{:.2f}",
            "Margin %": "{:.1f}%"
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Show More button
    show_more_button(menu_key, total_menu_items)

    st.divider()

    # Live Menu Preview with Design Options
    st.subheader("[Design] Menu Designer & Exporter")

    # Menu design style selector
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        menu_style = st.selectbox(
            "Select Design Style",
            ["Classic Elegance", "Modern Minimalist", "Bold & Vibrant"],
            help="Choose your preferred menu design style"
        )

    with col2:
        show_preview = st.checkbox("Show Live Preview", value=True)

    with col3:
        preview_height = st.slider("Preview Height", 400, 1200, 800, step=100, help="Adjust preview window height")

    st.divider()

    # Download options
    st.subheader("[Download] Download Options")

    # Generate menu files based on selected style
    styled_menu_html = generate_menu_html_styled(filtered_items, menu_style)
    menu_text = generate_menu_text(filtered_items)
    menu_csv = filtered_items[["item_title", "avg_price", "margin_percentage", "category"]].to_csv(index=False)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "[HTML] Download HTML Menu",
            styled_menu_html,
            f"menumetrics_menu_{menu_style.lower().replace(' ', '_')}.html",
            "text/html",
            use_container_width=True,
            help="Download styled HTML menu - open in browser and print to PDF"
        )

    with col2:
        st.download_button(
            "[Text] Download Text Version",
            menu_text,
            "menu_text.txt",
            "text/plain",
            use_container_width=True,
        )

    with col3:
        st.download_button(
            "[Data] Download Data (CSV)",
            menu_csv,
            "menu_data.csv",
            "text/csv",
            use_container_width=True,
        )

    # PDF Instructions
    st.info("[Tip] **To create a PDF:** Download the HTML file, open it in your browser (Chrome/Edge recommended), and use **Ctrl+P** or **Cmd+P** -> **Save as PDF**")

    # Live preview
    if show_preview:
        st.divider()
        st.markdown("### [Preview] Live Preview")
        st.markdown(f"*Preview of {menu_style} design - scroll to see full menu*")
        st.components.v1.html(styled_menu_html, height=preview_height, scrolling=True)


def generate_menu_html_styled(items_df: pd.DataFrame, style: str = "Classic Elegance") -> str:
    """
    Generate a styled HTML menu with multiple design options.

    Args:
        items_df (pd.DataFrame): Filtered menu items
        style (str): Design style - "Classic Elegance", "Modern Minimalist", or "Bold & Vibrant"

    Returns:
        str: HTML markup for the styled menu
    """
    if style == "Modern Minimalist":
        return generate_modern_menu(items_df)
    elif style == "Bold & Vibrant":
        return generate_vibrant_menu(items_df)
    else:
        return generate_classic_menu(items_df)


def generate_classic_menu(items_df: pd.DataFrame) -> str:
    """Generate a classic elegant menu design."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MenuMetrics - Classic Elegance</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400&display=swap');

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Lato', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 40px 20px;
            line-height: 1.8;
        }

        .menu-page {
            max-width: 900px;
            margin: 0 auto;
            background: #ffffff;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            border-radius: 8px;
            overflow: hidden;
        }

        .menu-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
            position: relative;
        }

        .menu-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120"><path d="M0,0 Q300,40 600,20 T1200,0 L1200,120 L0,120 Z" fill="rgba(255,255,255,0.1)"/></svg>') bottom center no-repeat;
            background-size: cover;
            opacity: 0.3;
        }

        h1 {
            font-family: 'Playfair Display', serif;
            font-size: 3.5em;
            font-weight: 700;
            margin-bottom: 15px;
            position: relative;
            z-index: 1;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .subtitle {
            font-size: 1.3em;
            font-weight: 300;
            letter-spacing: 2px;
            position: relative;
            z-index: 1;
            opacity: 0.95;
        }

        .menu-content {
            padding: 50px 60px;
        }

        .category-section {
            margin-bottom: 50px;
        }

        .category-header {
            display: flex;
            align-items: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }

        .category-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.2em;
            color: #2d3748;
            font-weight: 700;
        }

        .menu-item {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 1px dotted #cbd5e0;
        }

        .menu-item:last-child {
            border-bottom: none;
        }

        .item-info {
            flex: 1;
            padding-right: 20px;
        }

        .item-name {
            font-size: 1.3em;
            color: #2d3748;
            font-weight: 600;
            margin-bottom: 5px;
        }

        .item-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75em;
            margin-left: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .item-price {
            font-size: 1.4em;
            color: #667eea;
            font-weight: 700;
            white-space: nowrap;
            font-family: 'Playfair Display', serif;
        }

        .menu-footer {
            background: #f7fafc;
            padding: 30px;
            text-align: center;
            color: #718096;
            font-size: 0.95em;
            border-top: 1px solid #e2e8f0;
        }

        @media print {
            body { background: white; padding: 0; }
            .menu-page { box-shadow: none; max-width: 100%; }
        }
    </style>
</head>
<body>
    <div class="menu-page">
        <div class="menu-header">
            <h1>MenuMetrics</h1>
            <div class="subtitle">Optimized Menu Experience</div>
        </div>
        <div class="menu-content">
"""

    category_config = {
        "star": {"title": "Star Chef's Recommendations", "badge": "BEST SELLER"},
        "plowhorse": {"title": "Star Customer Favorites", "badge": "POPULAR"},
        "puzzle": {"title": "[Target] Specialty Items", "badge": "UNIQUE"}
    }

    for category, config in category_config.items():
        cat_items = items_df[items_df["category"] == category]
        if len(cat_items) == 0:
            continue

        html += f'            <div class="category-section">\n'
        html += f'                <div class="category-header">\n'
        html += f'                    <div class="category-title">{config["title"]}</div>\n'
        html += f'                </div>\n'

        for _, item in cat_items.head(15).iterrows():  # Limit to 15 items per category
            badge = f'<span class="item-badge">{config["badge"]}</span>' if category == "star" else ""
            html += f'                <div class="menu-item">\n'
            html += f'                    <div class="item-info">\n'
            html += f'                        <div class="item-name">{item["item_title"]}{badge}</div>\n'
            html += f'                    </div>\n'
            html += f'                    <div class="item-price">DKK {item["avg_price"]:.2f}</div>\n'
            html += f'                </div>\n'

        html += '            </div>\n'

    html += f"""
        </div>
        <div class="menu-footer">
            Optimized by MenuMetrics Intelligence Platform | {datetime.now().strftime('%B %Y')}
        </div>
    </div>
</body>
</html>
"""
    return html


def generate_modern_menu(items_df: pd.DataFrame) -> str:
    """Generate a modern minimalist menu design."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MenuMetrics - Modern</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', sans-serif;
            background: #0f0f0f;
            padding: 30px;
        }

        .menu-container {
            max-width: 1000px;
            margin: 0 auto;
            background: #1a1a1a;
            border: 1px solid #333;
        }

        .header {
            padding: 80px 60px;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border-bottom: 2px solid #00ff88;
        }

        h1 {
            font-size: 4em;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -2px;
            margin-bottom: 10px;
        }

        .tagline {
            color: #00ff88;
            font-size: 1.1em;
            font-weight: 300;
            letter-spacing: 3px;
            text-transform: uppercase;
        }

        .content {
            padding: 60px;
        }

        .section {
            margin-bottom: 60px;
        }

        .section-title {
            font-size: 1.8em;
            color: #00ff88;
            font-weight: 600;
            margin-bottom: 30px;
            text-transform: uppercase;
            letter-spacing: 2px;
            border-left: 4px solid #00ff88;
            padding-left: 20px;
        }

        .item {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 30px;
            padding: 20px 0;
            border-bottom: 1px solid #2d2d2d;
        }

        .item:hover {
            background: rgba(0,255,136,0.05);
            padding-left: 20px;
            transition: all 0.3s ease;
        }

        .item-name {
            font-size: 1.3em;
            color: #ffffff;
            font-weight: 600;
        }

        .item-price {
            font-size: 1.3em;
            color: #00ff88;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }

        .badge {
            display: inline-block;
            background: #00ff88;
            color: #0f0f0f;
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 0.65em;
            margin-left: 12px;
            font-weight: 700;
            letter-spacing: 1px;
        }

        .footer {
            padding: 40px 60px;
            background: #0f0f0f;
            color: #666;
            text-align: center;
            font-size: 0.9em;
            border-top: 1px solid #2d2d2d;
        }

        @media print {
            body { background: white; }
            .menu-container { background: white; border: none; }
            .header { background: #f0f0f0; }
            h1, .item-name { color: #000; }
            .tagline, .section-title, .item-price { color: #333; }
        }
    </style>
</head>
<body>
    <div class="menu-container">
        <div class="header">
            <h1>FLAVORCRAFT</h1>
            <div class="tagline">Optimized Menu</div>
        </div>
        <div class="content">
"""

    categories = {
        "star": {"title": "Featured Selection", "badge": "Star"},
        "plowhorse": {"title": "Popular Choices", "badge": ""},
        "puzzle": {"title": "Signature Dishes", "badge": ""}
    }

    for cat_key, cat_info in categories.items():
        items = items_df[items_df["category"] == cat_key]
        if len(items) == 0:
            continue

        html += f'            <div class="section">\n'
        html += f'                <div class="section-title">{cat_info["title"]}</div>\n'

        for _, item in items.head(15).iterrows():
            badge = f'<span class="badge">{cat_info["badge"]}</span>' if cat_key == "star" else ""
            html += f'                <div class="item">\n'
            html += f'                    <div class="item-name">{item["item_title"]}{badge}</div>\n'
            html += f'                    <div class="item-price">{item["avg_price"]:.2f} DKK</div>\n'
            html += f'                </div>\n'

        html += '            </div>\n'

    html += f"""
        </div>
        <div class="footer">
            Menu Intelligence System | Updated {datetime.now().strftime('%B %Y')}
        </div>
    </div>
</body>
</html>
"""
    return html


def generate_vibrant_menu(items_df: pd.DataFrame) -> str:
    """Generate a bold and vibrant menu design."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MenuMetrics - Vibrant</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
        }

        .menu {
            max-width: 950px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 30px 80px rgba(0,0,0,0.3);
        }

