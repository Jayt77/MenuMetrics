# MenuMetrics Intelligence Platform

Data-driven menu engineering toolkit for restaurants. Ingests historical order data, classifies items (BCG Matrix), surfaces pricing opportunities, and ships a Streamlit dashboard plus CLI export.

## What it does
- Classify menu items into Stars / Plowhorses / Puzzles / Dogs using profitability and popularity medians.
- Compute contribution margins, popularity scores, and satisfaction scores (rating � log(votes+1)).
- Recommend actions and price hints; build an optimization plan and menu generator.
- Deliver an interactive Streamlit dashboard and a CLI that exports insights to CSV.

## Tech stack
- Python 3.10+; Streamlit, Pandas, NumPy, SciPy, scikit-learn, Plotly.
- Optional: Groq API for LLM assistant (disabled without `GROQ_API_KEY`).

## Repo layout
```
README.md
requirements.txt
package.json           # Present for compliance; Node is not required to run the app
src/
  main.py              # CLI + dashboard launcher
  streamlit_dashboard.py
  models/ (data_loader.py, menu_engineering_models.py)
  services/ (menu_engineering_service.py, inventory_service.py)
  utils/ (helpers.py)
  api/ (routes.py)
tests/ (pytest suite placeholder: test_helpers.py)
docs/ (architecture/method notes; screenshots placeholders)
config/ (sample.env, README)
data/ (sample CSVs from Menu Engineering Parts 1 & 2)
```

## Installation
1) Python environment
```
python -m venv .venv
. .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```
2) Data files: ensure `data/Menu Engineering Part 1` and `data/Menu Engineering Part 2` exist (already included in repo snapshot). Also the AI assistant requires the `../order_timeline_data.csv` file for additional capabilities and functionality. It can be found in the repo's github releases.
3) (Optional) set environment variables by copying `config/sample.env` to `.env` and adjusting paths/keys.

## Usage
- Dashboard (recommended):
```
streamlit run src/streamlit_dashboard.py --theme.base=light
```
- CLI batch export:
```
python src/main.py --cli --output menu_insights.csv \
  --menu-items "Menu Engineering Part 2/dim_menu_items.csv" \
  --order-items "Menu Engineering Part 1/fct_order_items.csv"
```
- Prebuilt helper scripts: `restart_dashboard.bat` (Windows) and `start_dashboard.sh` (Linux/macOS), both configured for light theme.

## Configuration
- Streamlit theme/server defaults: `.streamlit/config.toml` (light theme enforced).
- App inputs: `.env` (copy from `config/sample.env`) or CLI flags.
- Data directory override: set `DATA_DIR` (default `data`).

## Tests
- Current suite: `tests/test_helpers.py` (utility functions).
- Run: `pytest -q`
- Add new tests under `tests/`; keep names `test_*.py`.

## Architecture (brief)
- `DataLoader`: ETL/normalization for menu and order CSVs; flexible column resolution.
- `MenuEngineeringService`: core analytics (BCG classification, pricing optimization, recommendations, behavior analysis, language signals) plus satisfaction scoring using ratings+votes.
- `streamlit_dashboard.py`: UI pages (Overview, Category Analysis, Pricing, Optimization Plan, Menu Generator, Language, Customer Insights, AI Assistant cache hooks).
- `main.py`: CLI entry; exports insights and pricing recommendations.

## Security & data
- No secrets committed. Keep API keys in `.env` (ignored by git).
- Data included is sample hackathon data; remove/replace for production.

## Team & contributions

- **Jayt** (9 commits)
  - `src/streamlit_dashboard.py` -- Built the main Streamlit dashboard UI and page routing
  - `src/services/menu_engineering_service.py` -- Implemented BCG classification, pricing optimization, and recommendation engine
  - `src/models/data_loader.py` -- Developed ETL pipeline and CSV normalization logic
  - `.streamlit/config.toml`, `requirements.txt` -- Project configuration and dependency management

- **georgenehma** (4 commits)
  - `src/models/menu_engineering_models.py` -- Defined data models for menu items and analytics outputs
  - `src/services/inventory_service.py` -- Built inventory tracking and cost analysis service
  - `src/utils/helpers.py` -- Created shared utility functions for formatting and calculations
  - `tests/test_helpers.py` -- Wrote unit tests for utility functions

- **selimalyyy** (3 commits)
  - `src/pages/landing_page.py` -- Designed and implemented the landing page layout
  - `src/adapters/universal_data_adapter.py` -- Built the universal data adapter for flexible CSV ingestion
  - `src/api/routes.py` -- Set up API route definitions
  - `config/sample.env`, `data/` -- Sample environment config and dataset preparation

- **yahiaelbanhawy** (3 commits)
  - `src/ai/gemini_assistant.py` -- Implemented the AI assistant powered by Gemini/Groq LLM integration
  - `src/models/user_model.py` -- Created user model for session and preference tracking
  - `src/main.py` -- Added CLI entry point and AI assistant wiring
  - `src/streamlit_dashboard.py` -- Integrated AI Assistant page into the dashboard

- **Moustafa Tanbouly** (non-technical)
  - UI/UX design direction and dashboard layout planning
  - Business requirements gathering and stakeholder analysis
  - Presentation design and project documentation

- **Saad El Chourbagei** (non-technical)
  - Market research and competitive analysis for menu engineering strategies
  - Business model validation and feature prioritization
  - Data collection and domain expertise on restaurant operations

  ##Screenshots and Overview

  ![WhatsApp Image 2026-02-07 at 11 51 18 PM](https://github.com/user-attachments/assets/5a3ad6db-981c-4675-b804-416aee5706eb)
![WhatsApp Image 2026-02-07 at 11 51 19 PM](https://github.com/user-attachments/assets/cc8a39c9-6add-471a-95bf-da5ba1efb5e5)
![WhatsApp Image 2026-02-07 at 11 51 20 PM](https://github.com/user-attachments/assets/24309b59-5b18-4420-8856-c2ac7d12396a)
![WhatsApp Image 2026-02-07 at 11 51 19 PM (1)](https://github.com/user-attachments/assets/b82af1f4-cb27-48cb-8ed0-435e057a61aa)


## How to contribute
1) Branch from `main` (e.g., `feat/pricing-elasticity`).
2) Commit with clear messages (`feat:`, `fix:`, `docs:`, `test:` prefixes encouraged).
3) Open PR; request review from another team member.

## License
MIT
