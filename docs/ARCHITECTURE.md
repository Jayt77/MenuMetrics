# Architecture Overview

## Flow
1) Data ingest: `DataLoader` reads menu/items CSVs (Parts 1 & 2), normalizes column names, estimates missing costs, filters invalid rows.
2) Analytics: `MenuEngineeringService` aggregates revenue/cost, computes popularity & profitability thresholds, merges satisfaction metrics (rating + votes), classifies items (BCG), builds recommendations and price hints.
3) Presentation: `streamlit_dashboard.py` surfaces pages (Overview, Category Analysis, Pricing, Optimization Plan, Menu Generator, Language, Customer Insights, AI Assistant cache hooks) and enforces light theme.
4) Export: `main.py --cli` writes insights CSV; dashboard offers downloads.

## Key modules
- `src/models/data_loader.py`: ETL and normalization utilities.
- `src/services/menu_engineering_service.py`: core analytics, recommendations, behavior and language analysis.
- `src/utils/helpers.py`: shared helpers used by services and tests.
- `src/streamlit_dashboard.py`: Streamlit UI + custom styling.

## Configuration
- `.env` (copy from `config/sample.env`) for file paths and optional `GROQ_API_KEY`.
- `.streamlit/config.toml` locks theme to light and sets port 8501.

## Testing
- Run `pytest -q` (see `tests/test_helpers.py` as a pattern). Add new suites per service to increase coverage.

## Deployment notes
- No Node runtime required despite `package.json` presence.
- Data paths assume `data/` relative to repo root; override with `DATA_DIR` env if needed.
