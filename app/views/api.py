from flask import Blueprint, jsonify, request
from flask_cors import CORS
from app.services.csv_service import insert_csv_data
from app.ML.cluster import handle_clustering
from app.services.csvhandler import extract_header, get_all_csv_data
from app.ML.random_forest import (
    load_data,
    explore_data,
    preprocess_data,
    split_data,
    train_model,
    evaluate_model,
    get_feature_importances,
)
from app.config import get_csv_file_path
from app.ML.linear_regression import (
    handle_linear_regression,
    handle_linear_regression_by_date)
import pandas as pd

api_bp = Blueprint('api', __name__)
CORS(api_bp)

items = []
csv_file_path = get_csv_file_path()


@api_bp.route('/csv', methods=['POST'])
def upload_csv():
    result = insert_csv_data()
    if result == "successful":
        return jsonify({"message": "CSV data successfully inserted."}), 200
    else:
        return jsonify({"error": result}), 500


@api_bp.route('/items', methods=['GET'])
def get_items():
    return jsonify(items)


@api_bp.route('/items', methods=['POST'])
def create_item():
    item = request.json
    items.append(item)
    return jsonify(item), 201


@api_bp.route('/items/<int:index>', methods=['PUT'])
def update_item(index):
    if index < 0 or index >= len(items):
        return jsonify({"error": "Item not found"}), 404
    items[index] = request.json
    return jsonify(items[index])


@api_bp.route('/items/<int:index>', methods=['DELETE'])
def delete_item(index):
    if index < 0 or index >= len(items):
        return jsonify({"error": "Item not found"}), 404
    deleted_item = items.pop(index)
    return jsonify(deleted_item)


@api_bp.route('/csv/data', methods=['GET'])
def get_all_data():
    all_data = get_all_csv_data(csv_file_path)
    return jsonify(all_data)


@api_bp.route('/csv/headers', methods=['GET'])
def extract_csv_header():
    headers = extract_header(csv_file_path)
    return jsonify(headers)


@api_bp.route('/cluster', methods=['POST'])
def cluster_data():
    params = request.json
    num_clusters = params.get('num_clusters', 3)
    features = params.get('features', ['latitude', 'longitude', 'price'])
    result = handle_clustering(csv_file_path, num_clusters, features)
    return jsonify(result)


@api_bp.route('/rf-load', methods=['GET'])
def load_rf_data():
    try:
        result = load_data(csv_file_path)
        result_json = result.to_dict(orient='records')
        return jsonify(result_json)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/rf-explore', methods=['GET'])
def explore_rf_data():
    try:
        df = load_data(csv_file_path)
        result = explore_data(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/rf-evaluate', methods=['GET'])
def evaluate_rf_data():
    try:
        result = evaluate_model(csv_file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/rf-feature-importances', methods=['GET'])
def rf_feature_importances():
    try:
        data = load_data(csv_file_path)
        features, target = preprocess_data(data)
        x_train, x_test, y_train, y_test = split_data(features, target)
        best_rf, _ = train_model(x_train, y_train)
        result = get_feature_importances(best_rf, features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/linear_regression', methods=['POST'])
def linear_regression():
    params = request.json
    independent_vars = params.get('independent_vars', [])
    dependent_var = params.get('dependent_var', '')
    result = handle_linear_regression(csv_file_path, independent_vars, dependent_var)
    return jsonify(result)


@api_bp.route('/linear_regression_by_date', methods=['POST'])
def linear_regression_by_date():
    params = request.json
    independent_vars = params.get('independent_vars', [])
    dependent_var = params.get('dependent_var', '')
    start_date = params.get('start_date', '')
    end_date = params.get('end_date', '')

    result = handle_linear_regression_by_date(csv_file_path, independent_vars, dependent_var, start_date, end_date)
    return jsonify(result)

