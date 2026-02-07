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


def show_landing_page() -> bool:
    """
    Display the landing page with data upload.

    Returns:
        True if data was successfully uploaded and processed
    """
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px 40px 20px;'>
        <h1 style='font-size: 4em; font-weight: 700; color: #1f2937; margin-bottom: 20px;'>
            Welcome to MenuMetrics
        </h1>
        <p style='font-size: 1.8em; color: #6b7280; font-weight: 300; margin-bottom: 40px;'>
            Get menu optimizations in minutes
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Value Proposition
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3 style='color: #4f46e5; margin-bottom: 15px;'>Data-Driven Insights</h3>
            <p>BCG Matrix analysis, profitability metrics, and performance indicators</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3 style='color: #4f46e5; margin-bottom: 15px;'>Instant Results</h3>
            <p>Upload your data and get actionable recommendations in seconds</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3 style='color: #4f46e5; margin-bottom: 15px;'>Smart Automation</h3>
            <p>Automatically detects and maps your data structure</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Data Upload Section
    st.markdown("""
    <div style='text-align: center; padding: 30px 20px 20px 20px;'>
        <h2 style='font-size: 2.5em; color: #1f2937;'>Get Started</h2>
        <p style='font-size: 1.2em; color: #6b7280; margin-bottom: 30px;'>
            Upload your restaurant data to begin analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Upload Options
    upload_col1, upload_col2 = st.columns([2, 1])

    with upload_col1:
        st.subheader("Option 1: Upload Your Own Data")

        st.info("Upload one or more CSV files containing your menu and transaction data")

        uploaded_files = st.file_uploader(
            "Upload CSV Files (you can select multiple files)",
            type=['csv'],
            accept_multiple_files=True,
            help="Select all CSV files you want to analyze. You can upload order data, menu items, payments, etc.",
            key="files_upload"
        )

        if uploaded_files and len(uploaded_files) > 0:
            st.write(f"**{len(uploaded_files)} file(s) uploaded:**")
            for file in uploaded_files:
                st.write(f"- {file.name}")

            if st.button("Process Uploaded Data", type="primary", use_container_width=True):
                success = process_uploaded_files(uploaded_files)
                if success:
                    import time
                    st.success("Data processed successfully! Redirecting to dashboard...")
                    st.balloons()
                    time.sleep(1.0)  # Show balloons animation
                    st.rerun()

    with upload_col2:
        st.subheader("Option 2: Use Sample Data")

        st.write("Try MenuMetrics with our sample restaurant dataset")

        if st.button("Load Sample Data", use_container_width=True):
            success = load_sample_data()
            if success:
                import time
                st.success("Sample data loaded! Redirecting to dashboard...")
                time.sleep(0.5)  # Brief pause to show success message
                st.rerun()

    st.markdown("---")

    # What We Analyze Section
    st.markdown("""
    <div style='padding: 40px 20px;'>
        <h2 style='text-align: center; font-size: 2.2em; color: #1f2937; margin-bottom: 40px;'>
            What You'll Get
        </h2>
    </div>
    """, unsafe_allow_html=True)

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown("""
        **Menu Engineering Analysis**
        - BCG Matrix classification (Stars, Plowhorses, Puzzles, Dogs)
        - Profitability and popularity metrics
        - Strategic recommendations per item

        **Pricing Optimization**
        - Price elasticity analysis
        - Optimal pricing suggestions
        - Revenue impact projections
        """)

    with feature_col2:
        st.markdown("""
        **Performance Insights**
        - Item-level revenue and profit analysis
        - Category performance comparison
        - Menu health scoring

        **Actionable Reports**
        - Downloadable optimization plans
        - Visual menu designs (3 professional styles)
        - Export-ready formats (HTML, CSV, PDF)
        """)

    st.markdown("---")

    # Data Requirements
    with st.expander("Data Requirements & Column Mapping"):
        st.markdown("""
        ### Required Data Fields

        **For Transaction/Order Data:**
        - `Item ID` or `Product ID` (any variation)
        - `Quantity` or `Units Sold`

        **Optional Fields (Recommended):**
        - `Price` or `Unit Price`
        - `Cost` or `COGS` (if not provided, estimated at 30% of price)
        - `Date` or `Timestamp`
        - `Customer ID` (for segmentation analysis)

        ### How It Works

        1. **Upload** your CSV file(s)
        2. **Auto-Detection** - System analyzes your column names
        3. **Mapping** - Columns are intelligently mapped to standard schema
        4. **Validation** - Data quality is checked
        5. **Processing** - Analysis runs automatically
        6. **Results** - View insights in the dashboard

        ### Supported Variations

        Our system recognizes many column name variations:
        - Item names: `item_name`, `product_name`, `dish_name`, `title`
        - Quantities: `qty`, `quantity`, `units`, `sold`
        - Prices: `price`, `unit_price`, `selling_price`
        - Costs: `cost`, `cogs`, `food_cost`, `ingredient_cost`

        **Don't see your column names? No problem!** The system uses fuzzy matching to find the best fits.
        """)

    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 40px 20px; color: #9ca3af;'>
        <p>Powered by MenuMetrics Intelligence Platform</p>
        <p style='font-size: 0.9em;'>Professional menu optimization for restaurants and food service businesses</p>
    </div>
    """, unsafe_allow_html=True)

    return False


def process_uploaded_files(uploaded_files) -> bool:
    """
    Process multiple uploaded CSV files.

    Args:
        uploaded_files: List of uploaded CSV files

    Returns:
        True if processing successful
    """
    try:
        with st.spinner("Analyzing your files..."):
            # Initialize adapter
            adapter = UniversalDataAdapter()

            all_dataframes = {}
            profiles = {}

            # Load and profile all files
            for file in uploaded_files:
                st.write(f"Processing {file.name}...")
                df = pd.read_csv(file)
                all_dataframes[file.name] = df

                # Profile the data
                profile = adapter.profile_dataset(df, file.name)
                profiles[file.name] = profile

                st.write(f"- {file.name}: {profile.row_count:,} rows, {profile.column_count} columns, Type: {profile.detected_type}")

            # Find the main transaction/order file (largest or has order-like columns)
            main_file = None
            main_df = None
            max_rows = 0

            for filename, df in all_dataframes.items():
                if len(df) > max_rows:
                    max_rows = len(df)
                    main_file = filename
                    main_df = df

            if main_df is None:
                st.error("No valid data files found")
                return False

            st.success(f"Using {main_file} as primary data source")

            # Profile and map the main file
            profile = profiles[main_file]
            mappings = adapter.map_columns(main_df, profile, target_schema='transaction_data')

            if not mappings:
                st.error("Could not automatically map columns. Please check your data format.")
                return False

            # Show mappings
            with st.expander("Column Mappings for " + main_file):
                for mapping in mappings:
                    st.write(f"✓ `{mapping.uploaded_name}` → `{mapping.standard_name}` "
                            f"(confidence: {mapping.confidence:.0%})")

            # Validate
            validation = adapter.validate_data(main_df, mappings)

            if validation.errors:
                st.error("Data validation failed:")
                for error in validation.errors:
                    st.error(f"- {error}")
                return False

            if validation.warnings:
                with st.expander("Warnings"):
                    for warning in validation.warnings:
                        st.warning(f"- {warning}")

            # Transform data
            transformed_df = adapter.transform_data(main_df, mappings)

            # Store in session state
            st.session_state.uploaded_data = transformed_df
            st.session_state.data_profile = profile
            st.session_state.data_mappings = mappings
            st.session_state.data_source = 'uploaded'
            st.session_state.all_uploaded_files = all_dataframes  # Store all files for reference

            return True

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False


def process_uploaded_data(orders_file, menu_file=None) -> bool:
    """
    Process uploaded data files using Universal Data Adapter.

    Args:
        orders_file: Uploaded orders CSV file
        menu_file: Optional uploaded menu CSV file

    Returns:
        True if processing successful
    """
    try:
        with st.spinner("Analyzing your data..."):
            # Initialize adapter
            adapter = UniversalDataAdapter()

            # Read orders file
            orders_df = pd.read_csv(orders_file)

            # Profile the data
            profile = adapter.profile_dataset(orders_df, "orders")

            # Display profile summary
            st.info(f"""
            **Data Profile:**
            - Rows: {profile.row_count:,}
            - Columns: {profile.column_count}
            - Type: {profile.detected_type.title()} Data
            - Quality Score: {profile.quality_score:.1f}/100
            - Completeness: {profile.completeness:.1f}%
            """)

            # Map columns
            mappings = adapter.map_columns(orders_df, profile, target_schema='transaction_data')

            if not mappings:
                st.error("Could not automatically map columns. Please check your data format.")
                return False

            # Show mappings
            with st.expander("Column Mappings"):
                for mapping in mappings:
                    st.write(f"✓ `{mapping.uploaded_name}` → `{mapping.standard_name}` "
                            f"(confidence: {mapping.confidence:.0%})")

            # Validate
            validation = adapter.validate_data(orders_df, mappings)

            if validation.errors:
                st.error("Data validation failed:")
                for error in validation.errors:
                    st.error(f"- {error}")
                return False

            if validation.warnings:
                st.warning("Warnings:")
                for warning in validation.warnings:
                    st.warning(f"- {warning}")

            if validation.recommendations:
                with st.expander("Recommendations for Better Analysis"):
                    for rec in validation.recommendations:
                        st.info(f"- {rec}")

            # Transform data
            transformed_df = adapter.transform_data(orders_df, mappings)

            # Store in session state
            st.session_state.uploaded_data = transformed_df
            st.session_state.data_profile = profile
            st.session_state.data_mappings = mappings
            st.session_state.data_source = 'uploaded'

            # If menu file provided, process it
            if menu_file:
                menu_df = pd.read_csv(menu_file)
                menu_profile = adapter.profile_dataset(menu_df, "menu")
                menu_mappings = adapter.map_columns(menu_df, menu_profile, target_schema='menu_master')
                transformed_menu = adapter.transform_data(menu_df, menu_mappings)
                st.session_state.uploaded_menu = transformed_menu

            return True

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False

