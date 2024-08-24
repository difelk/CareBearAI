import os
import pickle

import numpy as np
from flask import Blueprint, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import io
from flask import Response, request, jsonify
from app.services.csv_service import insert_csv_data
from app.ML.cluster import handle_clustering
from sklearn.cluster import KMeans
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
    if categories:
        categories = [categories] if isinstance(categories, str) else categories
        df = df[df['category'].isin(categories)]
    if commodities:
        commodities = [commodities] if isinstance(commodities, str) else commodities
        df = df[df['commodity'].isin(commodities)]
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

    if not dataset or not linearXaxis or not linearYaxis:
        return jsonify({'error': 'Missing required parameters.'}), 400

    result = get_linear_regression(dataset, linearXaxis, linearYaxis)

    return jsonify(result)


@api_bp.route('/modals/rf-evaluate', methods=['POST'])
def evaluate_rf_data():
    try:

        params = request.json
        df = pd.json_normalize(params)

        results = evaluate_model(df)

        if isinstance(results, dict):

            return jsonify(results)
        else:

            return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/rf-forecast-custom', methods=['POST'])
def forecast_custom_rf():
    try:
        params = request.json
        dataset = params.get('dataset')

        df = pd.DataFrame(dataset)

        filtered_df = filter_data(df, params.get('market'), params.get('category'), params.get('commodity'))

        if filtered_df.empty:
            return jsonify({"error": "No data found for the specified filters"}), 404

        features, target = preprocess_data(filtered_df)
        x_train, x_test, y_train, y_test = split_data(features, target)
        model, _ = train_model(x_train, y_train)

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

        params = request.json
        df = pd.json_normalize(params)

        results = svm_evaluate_model(df)

        if isinstance(results, dict):

            return jsonify(results)
        else:

            return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/svm-forecast-custom', methods=['POST'])
def forecast_custom_svm():
    try:
        params = request.json
        dataset = params.get('dataset')

        df = pd.DataFrame(dataset)

        filtered_df = filter_data(df, params.get('market'), params.get('category'), params.get('commodity'))

        if filtered_df.empty:
            return jsonify({"error": "No data found for the specified filters"}), 404

        features, target, scaler = svm_preprocess_data(filtered_df)
        x_train, x_test, y_train, y_test = svm_split_data(features, target)
        model, _, scaler, feature_columns, training_dtypes = svm_train_model(x_train, y_train)

        forecasts = {}
        unique_commodities = filtered_df['commodity'].unique()
        for commodity in unique_commodities:
            forecast_df = svm_forecast_prices(filtered_df, model, commodity=commodity)
            forecasts[commodity] = forecast_df.to_dict(orient='records')

        return jsonify({"forecasts": forecasts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/forecast-high-low', methods=['POST'])
def forecast_prices_svm():
    params = request.json
    dataset = params.get('dataset')
    end_date_str = params.get('end_date')
    start_date_str = params.get('start_date')

    if not end_date_str or not start_date_str:
        return jsonify({"error": "Please provide start_date and end_date in the request body."}), 400

    try:
        end_date = pd.to_datetime(end_date_str)
        start_date = pd.to_datetime(start_date_str)
    except Exception as e:
        return jsonify({"error": "Invalid date format. Ensure dates are in the correct format."}), 400

    if end_date is None:
        end_date = datetime.now()

    months = 1
    start_date = end_date - timedelta(days=30 * months)

    df = pd.json_normalize(dataset)

    if 'date' not in df.columns:
        return jsonify({"error": "Dataset must contain a 'date' column."}), 400

    df['date'] = pd.to_datetime(df['date'])

    historical_averages = get_historical_averages(df, end_date)

    filtered_data = prepare_forecast_data(df, historical_averages)

    features, target, scaler = svm_preprocess_data(filtered_data)
    x_train, x_test, y_train, y_test = svm_split_data(features, target)

    model, _, scaler, feature_columns, training_dtypes = svm_train_model(x_train, y_train)

    if model is None:
        return jsonify({"error": "Model is not trained yet."}), 500

    forecasted_classes = svm_forecast_price_class(model, filtered_data, feature_columns, training_dtypes, scaler)

    response = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "forecasts": forecasted_classes
    }

    return jsonify(response)


@api_bp.route('/modals/linear_regression/risk_management', methods=['POST'])
def get_risk_management():
    try:
        params = request.json
        dataset = params.get('dataset')

        #dataset to dataFrame
        df = pd.DataFrame(dataset)

        # data cleaning
        df = df.dropna()
        df = df.drop_duplicates()
        #convert relevant columns to appropriate types
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # remove the data row that covert failed

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        df = df.dropna()

        features, target = lr_preprocess_data(df)

        # splitting
        x_train, x_test, y_train, y_test = lr_split_data(features, target)

        # training
        model = lr_train_model(x_train, y_train)

        # Predictions
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


@api_bp.route('/modals/linear_regression/visualization_data', methods=['POST'])
def get_visualization_data():
    try:

        params = request.get_json()
        commodity = params.get('commodity')
        market = params.get('market')
        category = params.get('category')
        data = pd.DataFrame(params.get('dataset'))

        features, target = lr_preprocess_data(data)
        x_train, x_test, y_train, y_test = lr_split_data(features, target)
        model = lr_train_model(x_train, y_train)

        if not hasattr(model, 'predict'):
            raise ValueError("Model is not correctly initialized.")

        future_dates, forecasted_prices = lr_forecast_prices(data, model, commodity, market, category)

        min_price = np.min(forecasted_prices)
        max_price = np.max(forecasted_prices)
        average_price = np.mean(forecasted_prices)
        trend_direction = "increasing" if forecasted_prices[-1] > forecasted_prices[0] else "decreasing"
        percentage_change = ((forecasted_prices[-1] - forecasted_prices[0]) / abs(forecasted_prices[0])) * 100

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

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/linear_regression/price_predictions', methods=['POST'])
def get_price_predictions():
    try:

        params = request.get_json()
        commodity = params.get('commodity')
        market = params.get('market')
        category = params.get('category')

        dataset = params.get('dataset')
        if dataset is None:
            return jsonify({"error": "Dataset is required"}), 400

        data = pd.DataFrame(dataset)

        if not isinstance(data, pd.DataFrame):
            return jsonify({"error": "Data is not in DataFrame format"}), 500

        features, target = lr_preprocess_data(data)
        x_train, x_test, y_train, y_test = lr_split_data(features, target)

        model = lr_train_model(x_train, y_train)

        future_dates, future_predictions = lr_forecast_prices(data, model, commodity, market, category, max_periods=120)

        if future_dates is None:
            return jsonify({"error": future_predictions}), 500

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
                continue

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


@api_bp.route('/modals/predictions/linear_regression/get-trend', methods=['POST'])
def get_trend():
    data = request.json
    df = pd.DataFrame(data)

    df['date'] = pd.to_datetime(df['date'])

    df['year_month'] = df['date'].dt.to_period('M').astype(str)

    monthly_avg = df.groupby('year_month')['price'].mean().reset_index()

    result = monthly_avg.to_dict(orient='records')

    return jsonify(result)


@api_bp.route('/modals/clustering/markets/price_levels', methods=['POST'])
def cluster_markets_by_price_levels():
    try:
        params = request.get_json()
        dataset = params.get('dataset')

        if not isinstance(dataset, list):
            return jsonify({"error": "Dataset should be a list of records"}), 400

        if not dataset:
            return jsonify({"error": "Dataset is empty"}), 400

        data = pd.DataFrame(dataset)
        logging.debug(f"Initial Data: {data.head()}")

        required_columns = ['commodity', 'category', 'market', 'price']
        if not all(col in data.columns for col in required_columns):
            return jsonify({"error": f"Dataset must contain the following columns: {', '.join(required_columns)}"}), 400

        commodities = data['commodity'].unique()
        categories = data['category'].unique()

        logging.debug(f"Detected Commodities: {commodities}")
        logging.debug(f"Detected Categories: {categories}")

        market_prices = data.groupby('market')['price'].mean().reset_index()
        logging.debug(f"Market Average Prices: {market_prices.head()}")

        kmeans = KMeans(n_clusters=3)
        market_prices['cluster'] = kmeans.fit_predict(market_prices[['price']])

        cluster_map = {0: 'Low Price', 1: 'Medium Price', 2: 'High Price'}
        market_prices['cluster_label'] = market_prices['cluster'].map(cluster_map)

        result = {
            "commodities": commodities.tolist(),
            "categories": categories.tolist(),
            "clusters": market_prices[['market', 'cluster_label']].to_dict(orient='records')
        }

        return jsonify(result)

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

        df = pd.DataFrame(dataset)

        preprocessed_data = km_preprocess_data(df)

        x_train, x_test = km_split_data(preprocessed_data)

        model = km_train_model(x_train)

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

        forecasted_clusters = km_forecast_clusters(preprocessed_data, model)

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

        visualized_data_dict = visualized_data.to_dict(orient='records')

        return jsonify(visualized_data_dict)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/modals/k-means/km-insights', methods=['POST'])
def cluster_insights():
    try:
        params = request.json

        if isinstance(params, list):
            dataset = params
        elif isinstance(params, dict):
            dataset = params.get('dataset')
        else:
            return jsonify({"error": "Invalid data format. Expected a dictionary or list."}), 400

        if not dataset:
            return jsonify({"error": "Dataset is required"}), 400

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


@api_bp.route('/modals/k-means/cluster-markets', methods=['POST'])
def cluster_markets():
    try:
        data = request.json

        if 'dataset' not in data:
            return jsonify({"error": "No dataset provided"}), 400

        df = pd.DataFrame(data['dataset'])

        if not {'latitude', 'longitude'}.issubset(df.columns):
            return jsonify({"error": "Missing required columns"}), 400

        # Drop rows with missing values in 'latitude' or 'longitude'
        df = df.dropna(subset=['latitude', 'longitude'])

        # Extract Latitude and Longitude for clustering
        X = df[['latitude', 'longitude']]

        # Perform K-Means Clustering
        k = data.get('k', 5)  # Default to 5 clusters if not specified
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # Prepare the response data
        clusters = []
        for _, row in df.iterrows():
            clusters.append({
                "latitude": row['latitude'],
                "longitude": row['longitude'],
                "cluster": int(row['Cluster'])
            })

        centroids = [{"latitude": float(lat), "longitude": float(lon), "cluster": int(i)}
                     for i, (lat, lon) in enumerate(kmeans.cluster_centers_)]

        return jsonify({"clusters": clusters, "centroids": centroids})

    except Exception as e:
        # Log the error
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


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


@api_bp.route('/plot/<plot_name>')
def serve_plot(plot_name):
    plot_path = f'/Users/ilmeedesilva/Desktop/ML Ass 4/careBareAI/CareBearAI/app/static/{plot_name}'
    if os.path.exists(plot_path):
        return send_file(plot_path)
    else:
        os.abort(404)
