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
from app.ML.svm import (
    svm_load_data,
    svm_explore_data,
    svm_preprocess_data,
    svm_split_data,
    svm_train_model,
    svm_evaluate_model,
    svm_generate_roc_curve,
    svm_generate_precision_recall_curve
)
import pandas as pd

api_bp = Blueprint('api', __name__)
CORS(api_bp)

items = []
csv_file_path = get_csv_file_path()


def filter_and_clean_data(file_path, start_date, end_date):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])

    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    cleaned_df = filtered_df.dropna()

    return cleaned_df


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


# rf-evaluate?start_date=2004-01-15&end_date=2004-05-15
@api_bp.route('/rf-evaluate', methods=['GET'])
def evaluate_rf_data():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not start_date or not end_date:
            return jsonify({"error": "Please provide both start_date and end_date"}), 400

        filtered_data = filter_and_clean_data(csv_file_path, start_date, end_date)

        result = evaluate_model(filtered_data)

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


@api_bp.route('/svm-load', methods=['GET'])
def load_svm_data():
    try:
        result = svm_load_data(csv_file_path)
        result_json = result.to_dict(orient='records')
        return jsonify(result_json)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/svm-explore', methods=['GET'])
def explore_svm_data():
    try:
        df = svm_load_data(csv_file_path)
        result = svm_explore_data(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/svm-evaluate', methods=['GET'])
def evaluate_svm_data():
    try:
        result = svm_evaluate_model(csv_file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/svm-roc', methods=['GET'])
def get_roc_curve():
    try:
        data = svm_load_data(csv_file_path)
        features, target = svm_preprocess_data(data)
        x_train, x_test, y_train, y_test = svm_split_data(features, target)
        best_svc, _ = svm_train_model(x_train, y_train)

        y_prob = best_svc.decision_function(x_test)

        result = svm_generate_roc_curve(y_test, y_prob)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/svm-precision-recall', methods=['GET'])
def get_precision_recall_curve():
    try:
        data = svm_load_data(csv_file_path)
        features, target = svm_preprocess_data(data)
        x_train, x_test, y_train, y_test = svm_split_data(features, target)
        best_svc, _ = svm_train_model(x_train, y_train)

        y_prob = best_svc.decision_function(x_test)

        result = svm_generate_precision_recall_curve(y_test, y_prob)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
