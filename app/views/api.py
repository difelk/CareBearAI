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
    forecast_prices,
    forecast_all_commodities,
    forecast_all_categories,
    forecast_all_markets,
    evaluate_forecast
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
from statsmodels.tsa.arima.model import ARIMA

api_bp = Blueprint('api', __name__)
CORS(api_bp)

items = []
csv_file_path = get_csv_file_path()


# def filter_and_clean_data(file_path, start_date, end_date):
#     df = pd.read_csv(file_path)
#     df['date'] = pd.to_datetime(df['date'])
#
#     filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
#
#     cleaned_df = filtered_df.dropna()
#
#     return cleaned_df

def filter_and_clean_data(data, start_date, end_date):
    # Check if the input is a file path or a DataFrame
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("data should be a file path or a DataFrame")

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


# forecast?start_date=2004-01-15&end_date=2005-05-15&commodity=Wheat%20flour
# @api_bp.route('/forecast', methods=['GET'])
# def get_forecast():
#     try:
#         start_date = request.args.get('start_date')
#         end_date = request.args.get('end_date')
#         commodity = request.args.get('commodity')
#
#         if not start_date or not end_date or not commodity:
#             return jsonify({"error": "Please provide start_date, end_date, and commodity"}), 400
#
#         # Load and filter data
#         filtered_data = filter_and_clean_data(csv_file_path, start_date, end_date)
#
#         # Generate forecast for the specified commodity
#         forecast = forecast_prices(filtered_data, commodity)
#
#         # Convert forecast to a list and return as JSON
#         forecast_list = forecast.tolist()
#         return jsonify({"commodity": commodity, "forecast": forecast_list})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @api_bp.route('/forecast-all-commos', methods=['GET'])
# def get_forecast_all_commodities():
#     try:
#         start_date = request.args.get('start_date')
#         end_date = request.args.get('end_date')
#
#         if not start_date or not end_date:
#             return jsonify({"error": "Please provide both start_date and end_date"}), 400
#
#         # Load and filter data
#         filtered_data = filter_and_clean_data(csv_file_path, start_date, end_date)
#
#         # Generate forecasts for all commodities
#         forecasts = forecast_all_commodities(filtered_data)
#
#         # Return forecasts as JSON
#         return jsonify({"forecasts": forecasts})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @api_bp.route('/forecast-market', methods=['GET'])
# def get_forecast_all_markets():
#     try:
#         start_date = request.args.get('start_date')
#         end_date = request.args.get('end_date')
#
#         if not start_date or not end_date:
#             return jsonify({"error": "Please provide both start_date and end_date"}), 400
#
#         # Load and filter data
#         filtered_data = filter_and_clean_data(csv_file_path, start_date, end_date)
#
#         # Generate forecasts for all commodities
#         forecasts = forecast_all_markets(filtered_data)
#
#         # Return forecasts as JSON
#         return jsonify({"forecasts": forecasts})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @api_bp.route('/forecast-category', methods=['GET'])
# def get_forecast_all_categories():
#     try:
#         start_date = request.args.get('start_date')
#         end_date = request.args.get('end_date')
#
#         if not start_date or not end_date:
#             return jsonify({"error": "Please provide both start_date and end_date"}), 400
#
#         # Load and filter data
#         filtered_data = filter_and_clean_data(csv_file_path, start_date, end_date)
#
#         # Generate forecasts for all commodities
#         forecasts = forecast_all_categories(filtered_data)
#
#         # Return forecasts as JSON
#         return jsonify({"forecasts": forecasts})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# # forecast-custom?start_date=2023-01-01&end_date=2023-12-31&market=Ampara&category=cereals%20and%20tubers
# # forecast-custom?start_date=2023-01-01&end_date=2023-12-31&market=Ampara&market=Colombo&commodity=Wheat&commodity=Rice
# # forecast-custom?start_date=2023-01-01&end_date=2023-12-31
# @api_bp.route('/forecast-custom', methods=['GET'])
# def forecast_custom():
#     try:
#         start_date = request.args.get('start_date')
#         end_date = request.args.get('end_date')
#         markets = request.args.getlist('market')  # Accepts multiple markets
#         categories = request.args.getlist('category')  # Accepts multiple categories
#         commodities = request.args.getlist('commodity')  # Accepts multiple commodities
#
#         if not start_date or not end_date:
#             return jsonify({"error": "Please provide start_date and end_date"}), 400
#
#         # Load and filter data
#         filtered_data = filter_and_clean_data(csv_file_path, start_date, end_date)
#
#         # Apply additional filters if provided
#         if markets:
#             filtered_data = filtered_data[filtered_data['market'].isin(markets)]
#         if categories:
#             filtered_data = filtered_data[filtered_data['category'].isin(categories)]
#         if commodities:
#             filtered_data = filtered_data[filtered_data['commodity'].isin(commodities)]
#
#         # Check if any data remains after filtering
#         if filtered_data.empty:
#             return jsonify({"error": "No data found for the specified filters"}), 404
#
#         # Initialize a dictionary to store forecasts
#         forecasts = {}
#
#         # Forecast for each commodity in the filtered data
#         unique_commodities = filtered_data['commodity'].unique()
#         for commodity in unique_commodities:
#             commodity_data = filtered_data[filtered_data['commodity'] == commodity]
#
#             # Ensure 'date' column is in datetime format and set as index
#             commodity_data['date'] = pd.to_datetime(commodity_data['date'])
#             commodity_data = commodity_data.set_index('date')
#
#             # Resample to monthly data and interpolate missing values
#             price_data = commodity_data['price'].resample('M').mean().interpolate()
#
#             # Fit ARIMA model
#             model = ARIMA(price_data, order=(1, 1, 1))  # Adjust order as needed
#             model_fit = model.fit()
#
#             # Forecast for the next 12 months
#             forecast = model_fit.forecast(steps=12)
#
#             # Store forecast in the dictionary
#             forecasts[commodity] = forecast.tolist()
#
#         return jsonify({"forecasts": forecasts})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


@api_bp.route('/forecast', methods=['GET'])
def get_forecast():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        commodity = request.args.get('commodity')
        market = request.args.get('market')
        category = request.args.get('category')

        if not start_date or not end_date:
            return jsonify({"error": "Please provide start_date and end_date"}), 400

        # Load, filter, and preprocess data
        filtered_data = filter_and_clean_data(csv_file_path, start_date, end_date)
        features, target = preprocess_data(filtered_data)
        x_train, x_test, y_train, y_test = split_data(features, target)
        model, _ = train_model(x_train, y_train)

        # Generate forecast
        forecast = forecast_prices(filtered_data, model, commodity=commodity, market=market, category=category)

        return jsonify({"forecast": forecast.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/forecast-all-commos', methods=['GET'])
def get_forecast_all_commodities():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not start_date or not end_date:
            return jsonify({"error": "Please provide both start_date and end_date"}), 400

        # Load, filter, and preprocess data
        filtered_data = filter_and_clean_data(csv_file_path, start_date, end_date)
        features, target = preprocess_data(filtered_data)
        x_train, x_test, y_train, y_test = split_data(features, target)
        model, _ = train_model(x_train, y_train)

        # Generate forecasts for all commodities
        forecasts = forecast_all_commodities(filtered_data, model)

        return jsonify({"forecasts": forecasts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/forecast-market', methods=['GET'])
def get_forecast_all_markets():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not start_date or not end_date:
            return jsonify({"error": "Please provide both start_date and end_date"}), 400

        # Load, filter, and preprocess data
        filtered_data = filter_and_clean_data(csv_file_path, start_date, end_date)
        features, target = preprocess_data(filtered_data)
        x_train, x_test, y_train, y_test = split_data(features, target)
        model, _ = train_model(x_train, y_train)

        # Generate forecasts for all markets
        forecasts = forecast_all_markets(filtered_data, model)

        return jsonify({"forecasts": forecasts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/forecast-category', methods=['GET'])
def get_forecast_all_categories():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not start_date or not end_date:
            return jsonify({"error": "Please provide both start_date and end_date"}), 400

        # Load, filter, and preprocess data
        filtered_data = filter_and_clean_data(csv_file_path, start_date, end_date)
        features, target = preprocess_data(filtered_data)
        x_train, x_test, y_train, y_test = split_data(features, target)
        model, _ = train_model(x_train, y_train)

        # Generate forecasts for all categories
        forecasts = forecast_all_categories(filtered_data, model)

        return jsonify({"forecasts": forecasts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# forecast-custom?start_date=2023-01-01&end_date=2023-12-31&market=Ampara&category=cereals%20and%20tubers
# forecast-custom?start_date=2023-01-01&end_date=2023-12-31&market=Ampara&market=Colombo&commodity=Wheat&commodity=Rice
# forecast-custom?start_date=2023-01-01&end_date=2023-12-31
@api_bp.route('/forecast-custom', methods=['GET'])
def forecast_custom():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        markets = request.args.getlist('market')
        categories = request.args.getlist('category')
        commodities = request.args.getlist('commodity')

        if not start_date or not end_date:
            return jsonify({"error": "Please provide start_date and end_date"}), 400

        # Load, filter, and preprocess data
        filtered_data = filter_and_clean_data(csv_file_path, start_date, end_date)
        features, target = preprocess_data(filtered_data)
        x_train, x_test, y_train, y_test = split_data(features, target)
        model, _ = train_model(x_train, y_train)

        # Apply additional filters if provided
        if markets:
            filtered_data = filtered_data[filtered_data['market'].isin(markets)]
        if categories:
            filtered_data = filtered_data[filtered_data['category'].isin(categories)]
        if commodities:
            filtered_data = filtered_data[filtered_data['commodity'].isin(commodities)]

        if filtered_data.empty:
            return jsonify({"error": "No data found for the specified filters"}), 404

        # Generate forecasts for filtered data
        forecasts = {}
        unique_commodities = filtered_data['commodity'].unique()
        for commodity in unique_commodities:
            forecast = forecast_prices(filtered_data, model, commodity=commodity)
            forecasts[commodity] = forecast.tolist()

        return jsonify({"forecasts": forecasts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



