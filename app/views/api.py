from flask import Blueprint, jsonify, request
from app.services.csv_service import insert_csv_data
from app.ML.cluster import handle_clustering

api_bp = Blueprint('api', __name__)

items = []

csv_file_path = r'C:\Users\Rananjaya\Desktop\ML\backend\app\dataset\wfp_food_prices_lka(wfp_food_prices_lka).csv'


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


# k-mean post
# api endpoint like this - http://127.0.0.1:5000/api/cluster
@api_bp.route('/cluster', methods=['POST'])
def cluster_data():
    # Get parameters from the request
    params = request.json
    num_clusters = params.get('num_clusters', 3)  # Default to 3 clusters
    features = params.get('features', ['latitude', 'longitude', 'price'])

    result = handle_clustering(csv_file_path, num_clusters, features)

    return jsonify(result)
