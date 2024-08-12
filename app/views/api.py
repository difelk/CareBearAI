from flask import Blueprint, jsonify, request
from app.services.csv_service import insert_csv_data

api_bp = Blueprint('api', __name__)

items = []


@api_bp.route('/csv', methods=['POST'])
def upload_csv():
    result = insert_csv_data()
    if result == "successful":
        return jsonify({"message": "CSV data successfully inserted."}), 200
    else:
        return jsonify({"error": result}), 500

# Get all items
@api_bp.route('/items', methods=['GET'])
def get_items():
    return jsonify(items)

# Create a new item
@api_bp.route('/items', methods=['POST'])
def create_item():
    item = request.json
    items.append(item)
    return jsonify(item), 201

# Update an item by index
@api_bp.route('/items/<int:index>', methods=['PUT'])
def update_item(index):
    if index < 0 or index >= len(items):
        return jsonify({"error": "Item not found"}), 404
    items[index] = request.json
    return jsonify(items[index])

# Delete an item by index
@api_bp.route('/items/<int:index>', methods=['DELETE'])
def delete_item(index):
    if index < 0 or index >= len(items):
        return jsonify({"error": "Item not found"}), 404
    deleted_item = items.pop(index)
    return jsonify(deleted_item)
