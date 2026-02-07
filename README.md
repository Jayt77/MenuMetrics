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
2) Data files: ensure `data/Menu Engineering Part 1` and `data/Menu Engineering Part 2` exist (already included in repo snapshot).
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
- Current git contributors (`git shortlog -sn --all`):
  - Jayt (3 commits)
- Action needed: add contributions from all team members per rubric; have each member commit their work with descriptive messages and feature branches.

## How to contribute
1) Branch from `main` (e.g., `feat/pricing-elasticity`).
2) Commit with clear messages (`feat:`, `fix:`, `docs:`, `test:` prefixes encouraged).
3) Open PR; request review from another team member.

## Known gaps vs requirements
- Single-contributor history; needs balanced commits by team.
- Tests minimal; expand coverage for services and dashboard logic.
- Screenshots are placeholders; replace with real captures from the running dashboard before submission.

## License
MIT
