from flask import Blueprint, jsonify, request

api_bp = Blueprint('api', __name__)

items = []

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
