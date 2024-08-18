from flask import Blueprint, jsonify, request
from flask_cors import CORS
import io
from flask import Response, request, jsonify
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
    forecast_prices,
    forecast_all_commodities,
    forecast_all_categories,
    forecast_all_markets,
    evaluate_forecast
)
from app.config import get_csv_file_path
from app.ML.linear_regression import (
    handle_linear_regression,
    handle_linear_regression_by_date,
    get_linear_regression)
from app.ML.svm import (
    svm_load_data,
    svm_explore_data,
    svm_preprocess_data,
    svm_split_data,
    svm_train_model,
    svm_evaluate_model,
    svm_generate_roc_curve,
    svm_generate_precision_recall_curve,
    svm_forecast_prices,
    svm_forecast_all_commodities,
    svm_forecast_all_categories,
    svm_forecast_all_markets,
    svm_forecast_price_class,
    get_historical_averages,
    prepare_forecast_data,

)
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

api_bp = Blueprint('api', __name__)
CORS(api_bp)

items = []
csv_file_path = get_csv_file_path()


def filter_data(df, markets, categories, commodities):
    if markets:
        markets = [markets] if isinstance(markets, str) else markets
        df = df[df['market'].isin(markets)]
        print("markets exist ", df)
    if categories:
        categories = [categories] if isinstance(categories, str) else categories
        df = df[df['category'].isin(categories)]
        print("categories exist ", df)
    if commodities:
        commodities = [commodities] if isinstance(commodities, str) else commodities
        df = df[df['commodity'].isin(commodities)]
        print("commodities exist ", df)
    return df


@api_bp.route('/csv', methods=['POST'])
def upload_csv():
    result = insert_csv_data()
    if result == "successful":
        return jsonify({"message": "CSV data successfully inserted."}), 200
    else:
        return jsonify({"error": result}), 500


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


@api_bp.route('/rf-explore', methods=['GET'])
def explore_rf_data():
    try:
        df = load_data(csv_file_path)
        result = explore_data(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/linear_regression', methods=['POST'])
def linear_regression():
    params = request.json

    dataset = params.get('dataset')
    linearXaxis = params.get('linearXaxis')
    linearYaxis = params.get('linearYaxis')

    # Validate that the required parameters are present
    if not dataset or not linearXaxis or not linearYaxis:
        return jsonify({'error': 'Missing required parameters.'}), 400

    result = get_linear_regression(dataset, linearXaxis, linearYaxis)

    return jsonify(result)


@api_bp.route('/modals/rf-evaluate', methods=['POST'])
def evaluate_rf_data():
    try:
        # Read JSON data and convert it to a DataFrame
        params = request.json
        df = pd.json_normalize(params)

        # Process the DataFrame with the evaluate_model function
        results = evaluate_model(df)

        # Check if the results are in dictionary form and convert to DataFrame if needed
        if isinstance(results, dict):
            # Directly return the dictionary as JSON
            return jsonify(results)
        else:
            # Assuming results are in list of dicts format
            return jsonify(results)  # Convert list of dicts to JSON

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/rf-forecast-custom', methods=['POST'])
def forecast_custom_rf():
    try:
        params = request.json
        dataset = params.get('dataset')

        # Convert dataset (which is a list of dictionaries) to DataFrame
        df = pd.DataFrame(dataset)
        print("Initial DataFrame:\n", df)

        # Apply filters to the DataFrame
        filtered_df = filter_data(df, params.get('market'), params.get('category'), params.get('commodity'))

        if filtered_df.empty:
            return jsonify({"error": "No data found for the specified filters"}), 404

        features, target = preprocess_data(filtered_df)
        x_train, x_test, y_train, y_test = split_data(features, target)
        model, _ = train_model(x_train, y_train)

        # Generate forecasts for filtered data
        forecasts = {}
        unique_commodities = filtered_df['commodity'].unique()
        for commodity in unique_commodities:
            forecast = forecast_prices(filtered_df, model, commodity=commodity)
            forecasts[commodity] = forecast.tolist()

        return jsonify({"forecasts": forecasts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/svm-evaluate', methods=['POST'])
def evaluate_svm_data():
    try:
        # Read JSON data and convert it to a DataFrame
        params = request.json
        df = pd.json_normalize(params)

        # Process the DataFrame with the evaluate_model function
        results = svm_evaluate_model(df)

        # Check if the results are in dictionary form and convert to DataFrame if needed
        if isinstance(results, dict):
            # Directly return the dictionary as JSON
            return jsonify(results)
        else:
            # Assuming results are in list of dicts format
            return jsonify(results)  # Convert list of dicts to JSON

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/svm-forecast-custom', methods=['POST'])
def forecast_custom_svm():
    try:
        params = request.json
        dataset = params.get('dataset')

        # Convert dataset (which is a list of dictionaries) to DataFrame
        df = pd.DataFrame(dataset)
        print("Initial DataFrame:\n", df)

        # Apply filters to the DataFrame
        filtered_df = filter_data(df, params.get('market'), params.get('category'), params.get('commodity'))

        if filtered_df.empty:
            return jsonify({"error": "No data found for the specified filters"}), 404

        features, target, scaler = svm_preprocess_data(filtered_df)
        x_train, x_test, y_train, y_test = svm_split_data(features, target)
        model, _, scaler, feature_columns, training_dtypes = svm_train_model(x_train, y_train)

        # Generate forecasts for filtered data
        forecasts = {}
        unique_commodities = filtered_df['commodity'].unique()
        for commodity in unique_commodities:
            forecast = svm_forecast_prices(filtered_df, model, commodity=commodity)
            forecasts[commodity] = forecast.tolist()

        return jsonify({"forecasts": forecasts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @api_bp.route('/forecast-high-low', methods=['GET'])
# def forecast_prices_svm():
#     start_date = request.args.get('start_date')
#     end_date = request.args.get('end_date')
#
#     if not start_date or not end_date:
#         return jsonify({"error": "Please provide start_date and end_date in the query parameters."}), 400
#
#     # Convert dates
#     end_date = pd.to_datetime(end_date)
#     start_date = pd.to_datetime(start_date)
#
#     # Filter and clean data
#
#     data = filter_and_clean_data(csv_file_path, start_date, end_date)
#     data['date'] = pd.to_datetime(data['date'])
#
#     # Get historical averages
#     historical_averages = get_historical_averages(data, end_date)
#
#     # Prepare forecast data with historical averages
#     filtered_data = prepare_forecast_data(data, historical_averages)
#
#     # Preprocess the data
#     features, target, scaler = svm_preprocess_data(filtered_data)
#     x_train, x_test, y_train, y_test = svm_split_data(features, target)
#
#     # Train the model
#     model, _, scaler, feature_columns, training_dtypes = svm_train_model(x_train, y_train)
#
#     if model is None:
#         return jsonify({"error": "Model is not trained yet."}), 500
#
#     # Forecast the prices
#     forecasted_classes = svm_forecast_price_class(model, filtered_data, feature_columns, training_dtypes, scaler)
#
#     response = {
#         "start_date": start_date,
#         "end_date": end_date,
#         "forecasts": forecasted_classes
#     }
#
#     return jsonify(response)


@api_bp.route('/modals/forecast-high-low', methods=['POST'])
def forecast_prices_svm():
    # start_date = request.args.get('start_date')
    # end_date = request.args.get('end_date')
    #
    # if not start_date or not end_date:
    #     return jsonify({"error": "Please provide start_date and end_date in the query parameters."}), 400
    #

    # Filter and clean data
    params = request.json
    dataset = params.get('dataset')
    end_date = params.get('end_date')
    start_date = params.get('start_date')

    # # Convert dates
    end_date = pd.to_datetime(end_date)
    start_date = pd.to_datetime(start_date)

    df = pd.json_normalize(dataset)

    # data = filter_and_clean_data(csv_file_path, start_date, end_date)
    df['date'] = pd.to_datetime(df['date'])

    # Get historical averages
    historical_averages = get_historical_averages(df, end_date)

    # Prepare forecast data with historical averages
    filtered_data = prepare_forecast_data(df, historical_averages)

    # Preprocess the data
    features, target, scaler = svm_preprocess_data(filtered_data)
    x_train, x_test, y_train, y_test = svm_split_data(features, target)

    # Train the model
    model, _, scaler, feature_columns, training_dtypes = svm_train_model(x_train, y_train)

    if model is None:
        return jsonify({"error": "Model is not trained yet."}), 500

    # Forecast the prices
    forecasted_classes = svm_forecast_price_class(model, filtered_data, feature_columns, training_dtypes, scaler)

    response = {
        "start_date": start_date,
        "end_date": end_date,
        "forecasts": forecasted_classes
    }

    return jsonify(response)
