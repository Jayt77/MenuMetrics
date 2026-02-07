@echo off
echo Stopping any running Streamlit instances...
taskkill //F //IM streamlit.exe 2>nul

echo.
echo Starting MenuMetrics Dashboard...
echo.
echo The dashboard will open in your browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

cd src
streamlit run streamlit_dashboard.py --server.port=8501 --theme.base=light
