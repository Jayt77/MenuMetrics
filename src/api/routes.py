"""
File: api_routes.py
Description: API endpoints for the application.
Dependencies: flask (or fastapi)
Author: Sample Team

This is a sample file demonstrating proper code structure and documentation.
Students should replace this with their actual implementation.

Note: This example uses Flask. Students can also use FastAPI, Express.js, or other frameworks.
"""

import os
from flask import Flask, request, jsonify
from typing import Dict, Any

from src.models.data_loader import DataLoader
from src.services.menu_engineering_service import MenuEngineeringService


app = Flask(__name__)


@app.route('/api/health', methods=['GET'])
def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify API is running.
    
    Returns:
        Dict[str, str]: Status message.
    
    Example Response:
        {
            "status": "healthy",
            "message": "API is running"
        }
    """
    return jsonify({
        "status": "healthy",
        "message": "API is running"
    })


@app.route('/api/inventory/predict', methods=['POST'])
def predict_inventory() -> Dict[str, Any]:
    """
    Predicts inventory demand for specified items.
    
    Request Body:
        {
            "item_id": "string",
            "period": "daily|weekly|monthly"
        }
    
    Returns:
        Dict[str, Any]: Prediction results.
    
    Example Response:
        {
            "item_id": "12345",
            "predicted_demand": 150.5,
            "period": "daily",
            "confidence": 0.85
        }
    
    Raises:
        400: If required parameters are missing.
        404: If item not found.
    """
    try:
        data = request.get_json()
        item_id = data.get('item_id')
        period = data.get('period', 'daily')
        
        if not item_id:
            return jsonify({"error": "item_id is required"}), 400
        
        # Students should implement actual prediction logic here
        # This is a placeholder response
        result = {
            "item_id": item_id,
            "predicted_demand": 150.5,
            "period": period,
            "confidence": 0.85,
            "timestamp": "2026-02-02T12:00:00Z"
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/menu/analyze', methods=['POST'])
def analyze_menu() -> Dict[str, Any]:
    """
    Analyzes menu items and provides recommendations.
    
    Request Body:
        {
            "place_id": "string",
            "analysis_type": "profitability|popularity|both"
        }
    
    Returns:
        Dict[str, Any]: Analysis results with categorized menu items.
    
    Example Response:
        {
            "stars": [...],
            "plowhorses": [...],
            "puzzles": [...],
            "dogs": [...]
        }
    """
    try:
        data = request.get_json()
        place_id = data.get("place_id")
        analysis_type = data.get("analysis_type", "both")
        menu_items_file = data.get(
            "menu_items_file",
            os.getenv("MENU_ITEMS_FILE", "Menu Engineering Part 1/dim_menu_items.csv"),
        )
        order_items_file = data.get(
            "order_items_file",
            os.getenv("MENU_ORDER_ITEMS_FILE", "Menu Engineering Part 2/fct_order_items.csv"),
        )
        data_dir = data.get("data_dir", os.getenv("DATA_DIR", "data"))

        if not place_id:
            return jsonify({"error": "place_id is required"}), 400

        loader = DataLoader(data_dir)
        service = MenuEngineeringService(loader)

        order_items = service.load_menu_data(menu_items_file, order_items_file, place_id)
        insights = service.build_menu_insights(order_items)

        categories = {
            "stars": "star",
            "plowhorses": "plowhorse",
            "puzzles": "puzzle",
            "dogs": "dog",
        }

        def build_payload(category_label: str) -> list:
            subset = insights[insights["category"] == category_label].head(10)
            return subset.to_dict(orient="records")

        result = {
            "place_id": place_id,
            "analysis_type": analysis_type,
            "summary": service.build_summary(insights),
            "stars": build_payload(categories["stars"]),
            "plowhorses": build_payload(categories["plowhorses"]),
            "puzzles": build_payload(categories["puzzles"]),
            "dogs": build_payload(categories["dogs"]),
        }

        return jsonify(result), 200

    except FileNotFoundError as exc:
        return jsonify({"error": f"Menu data files not found: {exc}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

