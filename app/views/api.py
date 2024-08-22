import os
import pickle

import numpy as np
from flask import Blueprint, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import io
from flask import Response, request, jsonify
from app.services.csv_service import insert_csv_data
from app.ML.cluster import handle_clustering
from app.services.csvhandler import extract_header, get_all_csv_data
from sklearn.linear_model import LinearRegression
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
    get_linear_regression,
    lr_split_data,
    lr_train_model,
    lr_preprocess_data,
    lr_forecast_prices,
    lr_load_data,
    lr_load_model,
    lr_explore_data,
    lr_evaluate_model,
    lr_save_results_to_json)
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
from app.ML.cluster import (km_visualize_clusters, km_forecast_clusters, km_cluster_insights, km_evaluate_model,
                            km_train_model, km_split_data, km_preprocess_data, km_load_data, km_explore_data,
                            interpret_forecasted_clusters)

import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

api_bp = Blueprint('api', __name__)
CORS(api_bp)

items = []
csv_file_path = get_csv_file_path()


def filter_data(df, markets, categories, commodities):
    if markets:
        markets = [markets] if isinstance(markets, str) else markets
        df = df[df['market'].isin(markets)]
        # print("markets exist ", df)
    if categories:
        categories = [categories] if isinstance(categories, str) else categories
        df = df[df['category'].isin(categories)]
        # print("categories exist ", df)
    if commodities:
        commodities = [commodities] if isinstance(commodities, str) else commodities
        df = df[df['commodity'].isin(commodities)]
        # print("commodities exist ", df)
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


# @api_bp.route('/modals/linear_regression', methods=['POST'])
# def linear_regression():
#     # Extract data from the request
#     data = request.json.get('dataset')
#     linearXaxis = request.json.get('linearXaxis')
#     linearYaxis = request.json.get('linearYaxis')
#
#     # Call the get_linear_regression function with the provided data
#     if not data or not linearXaxis or not linearYaxis:
#         return jsonify({'error': 'Invalid input data'}), 400
#
#     results = get_linear_regression(data, linearXaxis, linearYaxis)
#
#     # Return the results as a JSON response
#     return jsonify(results), 200


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


# @api_bp.route('/modals/rf-forecast-custom', methods=['POST'])
# def forecast_custom_rf():
#     try:
#         params = request.json
#         dataset = params.get('dataset')
#
#         # Convert dataset (which is a list of dictionaries) to DataFrame
#         df = pd.DataFrame(dataset)
#         print("Initial DataFrame:\n", df)
#
#         # Apply filters to the DataFrame
#         filtered_df = filter_data(df, params.get('market'), params.get('category'), params.get('commodity'))
#
#         if filtered_df.empty:
#             return jsonify({"error": "No data found for the specified filters"}), 404
#
#         features, target = preprocess_data(filtered_df)
#         x_train, x_test, y_train, y_test = split_data(features, target)
#         model, _ = train_model(x_train, y_train)
#
#         # Generate forecasts for filtered data
#         forecasts = {}
#         unique_commodities = filtered_df['commodity'].unique()
#         for commodity in unique_commodities:
#             forecast = forecast_prices(filtered_df, model, commodity=commodity)
#             forecasts[commodity] = forecast.tolist()
#
#         return jsonify({"forecasts": forecasts})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

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
            forecast_df = forecast_prices(filtered_df, commodity=commodity, market=None, category=None)
            forecasts[commodity] = forecast_df.reset_index().to_dict(orient='records')

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
            forecast_df = svm_forecast_prices(filtered_df, model, commodity=commodity)
            forecasts[commodity] = forecast_df.to_dict(orient='records')

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
    # Get parameters from request JSON
    params = request.json
    dataset = params.get('dataset')
    end_date_str = params.get('end_date')
    start_date_str = params.get('start_date')

    # Check if the required parameters are provided
    if not end_date_str or not start_date_str:
        return jsonify({"error": "Please provide start_date and end_date in the request body."}), 400

    # Convert dates from string to datetime
    try:
        end_date = pd.to_datetime(end_date_str)
        start_date = pd.to_datetime(start_date_str)
    except Exception as e:
        return jsonify({"error": "Invalid date format. Ensure dates are in the correct format."}), 400

    # Ensure end_date is not None and assign default if necessary
    if end_date is None:
        end_date = datetime.now()

    # Calculate the start date based on end date and months (assuming months is a parameter or constant)
    months = 1  # Set this according to your application logic
    start_date = end_date - timedelta(days=30 * months)

    # Convert dataset to DataFrame
    df = pd.json_normalize(dataset)

    # Check if 'date' column exists
    if 'date' not in df.columns:
        return jsonify({"error": "Dataset must contain a 'date' column."}), 400

    # Ensure 'date' column is in datetime format
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
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "forecasts": forecasted_classes
    }

    return jsonify(response)


# VISUALIZATION FOR RISK MANAGEMENT

# 1. Time Series Plot
# Purpose:
# To visualize how well your model's predictions match the actual prices over time.
# Data Needed:
# dates: The time periods corresponding to the predictions and actual values.
# actual_prices: The true prices from your dataset.
# predicted_prices: The values predicted by your model.
# Steps:
# Prepare Data:
#
# Ensure dates, actual_prices, and predicted_prices are in the same length and aligned correctly.
# Create Chart:
#
# Plot dates on the x-axis.
# Plot actual_prices and predicted_prices on the y-axis.
# Chart Type:
#
# Line Chart.

# Confidence Interval Plot
# Purpose: To visualize the range of uncertainty around predictions.
#
# What You Need:
#
# Dates (X-axis)
# Predicted Prices (Y-axis)
# Lower Bound of Confidence Interval (Y-axis)
# Upper Bound of Confidence Interval (Y-axis)
# How to Implement:
#
# Create a Line Chart with Shaded Area
# Plot the predicted prices as a line.
# Add the confidence intervals as shaded areas around the line.

@api_bp.route('/modals/linear_regression/risk_management', methods=['POST'])
def get_risk_management():
    try:
        params = request.json
        dataset = params.get('dataset')
        df = pd.DataFrame(dataset)
        features, target = lr_preprocess_data(df)
        x_train, x_test, y_train, y_test = lr_split_data(features, target)
        model = lr_train_model(x_train, y_train)

        y_pred = model.predict(x_test)

        return jsonify({
            "risk_management": {
                "dates": pd.to_datetime(df.iloc[x_test.index]['date']).tolist(),
                "actual_prices": y_test.tolist(),
                "predicted_prices": y_pred.tolist(),
                "confidence_intervals": {
                    "lower_bound": [np.percentile(y_pred, 2.5)] * len(y_pred),
                    "upper_bound": [np.percentile(y_pred, 97.5)] * len(y_pred)
                }
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Visualizations Breakdown

# Line Chart:
#
# Data: forecast_dates (x-axis) and forecast_prices (y-axis). Visualization: A simple line chart showing the
# predicted price trend over time. Implementation: Use libraries like Chart.js, D3.js, or any other charting library
# to plot the forecast_dates and forecast_prices. Summary Panel:
#
# Data: insights object. Visualization: Display min_price, max_price, average_price, trend_direction,
# and percentage_change as key metrics. Implementation: Create a summary panel or dashboard section that shows these
# insights as text or key performance indicators (KPIs). Trend Analysis:
#
# Data: trend_direction and percentage_change. Visualization: Indicate whether the price is generally increasing or
# decreasing, with an optional alert if the percentage change is significant. Implementation: Use icons (e.g.,
# arrows) or color-coding (green for increasing, red for decreasing) to visually represent the trend. The percentage
# change can be displayed alongside the trend direction for more context.
@api_bp.route('/modals/linear_regression/visualization_data', methods=['POST'])
def get_visualization_data():
    try:
        # Parse the incoming JSON data
        params = request.get_json()
        commodity = params.get('commodity')
        market = params.get('market')
        category = params.get('category')
        data = pd.DataFrame(params.get('dataset'))

        # Preprocess data
        features, target = lr_preprocess_data(data)
        x_train, x_test, y_train, y_test = lr_split_data(features, target)
        model = lr_train_model(x_train, y_train)

        # Ensure the model is a LinearRegression instance
        if not hasattr(model, 'predict'):
            raise ValueError("Model is not correctly initialized.")

        # Generate future dates and price predictions
        future_dates, forecasted_prices = lr_forecast_prices(data, model, commodity, market, category)

        # Calculate insights
        min_price = np.min(forecasted_prices)
        max_price = np.max(forecasted_prices)
        average_price = np.mean(forecasted_prices)
        trend_direction = "increasing" if forecasted_prices[-1] > forecasted_prices[0] else "decreasing"
        percentage_change = ((forecasted_prices[-1] - forecasted_prices[0]) / abs(forecasted_prices[0])) * 100

        # Prepare the response payload
        response_data = {
            "forecast_dates": future_dates.tolist(),
            "forecast_prices": forecasted_prices.tolist(),
            "insights": {
                "min_price": min_price,
                "max_price": max_price,
                "average_price": average_price,
                "trend_direction": trend_direction,
                "percentage_change": percentage_change,
            }
        }

        # Return the JSON response
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Line Chart:
#
# Purpose: To show how prices evolve over time.
# Values: forecast_dates (x-axis) vs. forecast_prices (y-axis).
# Bar Chart:
#
# Purpose: To compare forecasted prices at different time points.
# Values: Use forecast_dates as categories and forecast_prices as values.
# Scatter Plot with Trend Line:
#
# Purpose: To analyze the distribution of forecasted prices and the overall trend.
# Values: forecast_dates (x-axis) vs. forecast_prices (y-axis) with a fitted trend line.
# Percentage Change Visualization:
#
# Purpose: To highlight the rate of change in prices.
# Values: Use a text or annotation on the chart to show percentage change from the start to the end of the forecast period.
@api_bp.route('/modals/linear_regression/price_predictions', methods=['POST'])
def get_price_predictions():
    try:
        # Parse the incoming JSON data
        params = request.get_json()
        commodity = params.get('commodity')
        market = params.get('market')
        category = params.get('category')

        # Ensure dataset is provided
        dataset = params.get('dataset')
        if dataset is None:
            return jsonify({"error": "Dataset is required"}), 400

        # Convert the dataset from list to DataFrame
        data = pd.DataFrame(dataset)

        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            return jsonify({"error": "Data is not in DataFrame format"}), 500

        # Preprocess data and split into train/test sets
        features, target = lr_preprocess_data(data)
        x_train, x_test, y_train, y_test = lr_split_data(features, target)

        # Train the model
        model = lr_train_model(x_train, y_train)

        # Forecast future prices with a maximum limit on periods
        future_dates, future_predictions = lr_forecast_prices(data, model, commodity, market, category, max_periods=120)

        # Handle case where date generation fails
        if future_dates is None:
            return jsonify({"error": future_predictions}), 500

        # Return predictions as JSON response
        return jsonify({
            "price_predictions": {
                "forecast_dates": future_dates.tolist(),
                "forecast_prices": future_predictions.tolist()
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import logging

logging.basicConfig(level=logging.DEBUG)
@api_bp.route('/modals/predictions/linear_regression/price_predictions', methods=['POST'])
def get_linear_price_predictions():
    try:
        params = request.get_json()
        commodity = params.get('commodity')
        market = params.get('market')
        category = params.get('category')

        dataset = params.get('dataset')
        if dataset is None:
            return jsonify({"error": "Dataset is required"}), 400

        data = pd.DataFrame(dataset)
        logging.debug(f"Initial Data: {data.head()}")

        if not any([commodity, market, category]):
            return jsonify({"error": "At least one filter parameter (commodity, market, or category) is required"}), 400

        if commodity:
            data = data[data['commodity'].isin(commodity)]
        if market:
            data = data[data['market'].isin(market)]
        if category:
            data = data[data['category'].isin(category)]

        logging.debug(f"Filtered Data: {data.head()}")

        if data.empty:
            return jsonify({"error": "No data available for the provided filters"}), 404

        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')

        results = {}
        for group_name in (commodity or market or category):
            group_data = data[data['category'] == group_name] if category else (
                data[data['commodity'] == group_name] if commodity else
                data[data['market'] == group_name]
            )

            logging.debug(f"Group Data for {group_name}: {group_data.head()}")

            if group_data.empty:
                continue  # Skip if no data available for this group

            X = group_data['date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            y = group_data['price'].values

            model = LinearRegression()
            model.fit(X, y)

            last_date = group_data['date'].max()
            next_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 7)]
            next_dates_ordinal = np.array([date.toordinal() for date in next_dates]).reshape(-1, 1)

            predictions = model.predict(next_dates_ordinal)

            results[group_name] = []
            for date, price in zip(next_dates, predictions):
                results[group_name].append({
                    "date": date.strftime('%Y-%m'),
                    "predicted_price": price
                })

        return jsonify(results)

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/k-means/km-evaluate', methods=['POST'])
def evaluate_model_km():
    try:
        params = request.json
        dataset = params.get('dataset')
        if not dataset:
            return jsonify({"error": "Dataset is required"}), 400

        # Convert dataset to DataFrame
        df = pd.DataFrame(dataset)

        # Preprocess data
        preprocessed_data = km_preprocess_data(df)

        # Split data
        x_train, x_test = km_split_data(preprocessed_data)

        # Train model
        model = km_train_model(x_train)

        # Evaluate model
        evaluation_results = km_evaluate_model(model, x_test)

        return jsonify(evaluation_results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/k-means/km-forecast', methods=['POST'])
def forecast_clusters():
    try:
        params = request.json
        dataset = params.get('dataset', [])
        if not dataset:
            return jsonify({"error": "Dataset is required"}), 400

        df = pd.DataFrame(dataset)
        preprocessed_data = km_preprocess_data(df)
        x_train, _ = km_split_data(preprocessed_data)
        model = km_train_model(x_train)

        # Forecast clusters using the preprocessed data and trained model
        forecasted_clusters = km_forecast_clusters(preprocessed_data, model)

        # Interpret the forecasted clusters
        interpretation = interpret_forecasted_clusters(forecasted_clusters)

        return jsonify({
            'forecasted_clusters': forecasted_clusters.tolist(),
            'interpretation': interpretation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/k-means/km-visualize', methods=['POST'])
def visualize_clusters():
    try:
        params = request.json

        if not params:
            return jsonify({"error": "Dataset is required"}), 400

        df = pd.DataFrame(params)
        preprocessed_data = km_preprocess_data(df)
        x_train, _ = km_split_data(preprocessed_data)
        model = km_train_model(x_train)

        visualized_data = km_visualize_clusters(preprocessed_data, model)

        # Convert the DataFrame to a dictionary (or list of dictionaries)
        visualized_data_dict = visualized_data.to_dict(orient='records')

        return jsonify(visualized_data_dict)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/k-means/km-insights', methods=['POST'])
def cluster_insights():
    try:
        params = request.json

        # If params is a list, assume it's the dataset directly
        if isinstance(params, list):
            dataset = params
        elif isinstance(params, dict):
            dataset = params.get('dataset')
        else:
            return jsonify({"error": "Invalid data format. Expected a dictionary or list."}), 400

        if not dataset:
            return jsonify({"error": "Dataset is required"}), 400

        # Check if the dataset is a list
        if not isinstance(dataset, list):
            return jsonify({"error": "Invalid dataset format. Expected a list of records."}), 400

        df = pd.DataFrame(dataset)
        preprocessed_data = km_preprocess_data(df)
        x_train, _ = km_split_data(preprocessed_data)
        model = km_train_model(x_train)

        visualized_data = km_visualize_clusters(preprocessed_data, model)
        insights = km_cluster_insights(visualized_data)
        return jsonify(insights)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 1. Visualize Cluster Characteristics
# a. Cluster Distribution
#
# Chart Type: Pie Chart or Bar Chart
# Data: Count of data points per cluster
# Purpose: Show how data points are distributed among different clusters.
# b. Mean and Median Prices per Cluster
#
# Chart Type: Bar Chart
# Data: Mean and median prices for each cluster
# Purpose: Compare average and median prices across clusters.
# c. Price Range per Cluster
#
# Chart Type: Box Plot or Range Bar Chart
# Data: Price range (minimum and maximum) for each cluster
# Purpose: Display the variability of prices within each cluster.
# 2. Visualize Commodity and Market Distribution
# a. Commodities per Cluster
#
# Chart Type: Word Cloud or Bar Chart
# Data: Most frequent commodities in each cluster
# Purpose: Highlight the common commodities within each cluster.
# b. Markets per Cluster
#
# Chart Type: Bar Chart or Map (if geographic data is available)
# Data: Most frequent markets in each cluster
# Purpose: Show where the data points of each cluster are concentrated geographically or by market.
# 3. Interactive Visualizations
# a. Scatter Plot with Clusters
#
# Chart Type: Scatter Plot
# Data: Points with x and y coordinates, colored by cluster
# Purpose: Visualize the spatial distribution of data points and how they are grouped into clusters.
# b. Cluster Details
#
# Chart Type: Table or List
# Data: Detailed list of commodities, markets, and price statistics for each cluster
# Purpose: Provide users with a detailed view of each cluster’s composition.
# 4. Interpretations
# a. Cluster Distribution
#
# Interpretation: Explain which clusters are more populated and if any cluster has significantly fewer points. This can indicate the relative importance or rarity of certain clusters.
# b. Mean and Median Prices
#
# Interpretation: Describe the pricing trends within each cluster. For example, clusters with higher mean prices might represent premium markets or commodities.
# c. Price Range
#
# Interpretation: Highlight clusters with broad or narrow price ranges. A wide range could indicate variability in prices, while a narrow range might suggest a more consistent market.
# d. Commodities and Markets
#
# Interpretation: Discuss the most common commodities and markets within each cluster. This can help identify what types of goods are prevalent in specific clusters and where they are most commonly sold.
# Example Visualization Steps
# Prepare Data for Visualization:
#
# Extract the necessary data from the clustering results (mean prices, median prices, price ranges, commodities, markets).
# Select Visualization Tools:
#
# Use libraries like D3.js, Chart.js, or Plotly for interactive and dynamic visualizations.
# Implement Charts:
#
# Create pie charts for cluster distribution.
# Use bar charts for mean and median prices.
# Implement box plots or range bars for price ranges.
# Generate word clouds or bar charts for commodities.
# Develop maps or bar charts for market distribution.
# Add Interactivity:
#
# Allow users to filter by clusters and view detailed statistics.
# Provide hover-over details for more information on each data point or cluster.
# Present Insights:
#
# Include text-based explanations alongside visualizations to help users understand what the charts and plots are showing.


STATIC_DIR = '/Users/ilmeedesilva/Desktop/ML Ass 4/careBareAI/CareBearAI/app/static'


@api_bp.route('/modals/rf/plots', methods=['GET'])
def get_plots():
    plot_files = [
        'outliers_plot.png',
        'top_feature_importances.png',
        'actual_vs_predicted.png',
        'residuals_vs_predicted.png',
        'residual_histogram.png',
        'cv_score_distribution.png',
    ]
    plot_urls = {plot_file: f'/static/{plot_file}' for plot_file in plot_files}
    return jsonify(plot_urls)


# @api_bp.route('/static/<path:filename>', methods=['GET'])
# def serve_static(filename):
#     return send_from_directory(STATIC_DIR, filename)


@api_bp.route('/plot/<plot_name>')
def serve_plot(plot_name):
    plot_path = f'/Users/ilmeedesilva/Desktop/ML Ass 4/careBareAI/CareBearAI/app/static/{plot_name}'
    if os.path.exists(plot_path):
        return send_file(plot_path)
    else:
        os.abort(404)
