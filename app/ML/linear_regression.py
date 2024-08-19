import pickle
import warnings

import pandas as pd
from flask import jsonify
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json

from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

# Suppress warnings
warnings.filterwarnings('ignore')


def lr_load_data(file_path):
    return pd.read_csv(file_path)


def lr_explore_data(df):
    return {
        "head": df.head().to_dict(orient='records'),
        "tail": df.tail().to_dict(orient='records'),
        "info": str(df.info()),  # Convert info output to string
        "description": df.describe().to_dict()
    }


def lr_preprocess_data(data, training_features=None):
    # Outlier removal
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
    outliers = (z_scores > 3).any(axis=1)
    data = data[~outliers]
    data = data.dropna()

    # Date processing
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        data = data.drop(columns=['date'])

    # Categorical encoding
    categorical_columns = ['admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'currency', 'priceflag',
                           'pricetype']
    data = pd.get_dummies(data, columns=[col for col in categorical_columns if col in data.columns])

    # Scaling
    scaler = MinMaxScaler()
    numerical_features = ['latitude', 'longitude', 'price', 'usdprice', 'USD RATE']
    for feature in numerical_features:
        if feature in data.columns:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
            data[feature] = scaler.fit_transform(data[[feature]])

    # Ensure consistency with training features
    if training_features:
        # Ensure all training features are present in the prediction data
        for col in training_features:
            if col not in data.columns:
                data[col] = 0  # Add missing columns with default value 0

        # Reorder columns to match training columns
        data = data[training_features]

    target = data['price'] if 'price' in data.columns else None
    features = data.drop(columns=['price'], errors='ignore')

    return features, target


def lr_split_data(features, target):
    return train_test_split(features, target, train_size=0.8, test_size=0.2, random_state=100)


def lr_train_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    model.feature_names_in_ = x_train.columns.tolist()
    return model


def lr_load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)


def lr_evaluate_model(df):
    # Preprocess the data
    features, target = lr_preprocess_data(df)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(features, target, train_size=0.8, test_size=0.2,
                                                        random_state=100)

    # Train the model
    model = lr_train_model(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)

    # Evaluation metrics
    results = {
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "cv_scores": cross_val_score(model, x_test, y_test, cv=5).tolist()
    }

    print(results)
    return results


def lr_save_results_to_json(results, file_path='results.json'):
    with open(file_path, 'w') as f:
        json.dump(results, f)

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


def lr_forecast_prices(df, model, commodity=None, market=None, category=None, max_periods=120):
    # Filter the dataframe based on the provided filters (market, category, commodity)
    filtered_df = filter_data(df, market, category, commodity)

    # Preprocess the filtered data
    features, _ = lr_preprocess_data(filtered_df)

    # Ensure features have the same columns as the model expects
    training_columns = model.feature_names_in_
    features = features.reindex(columns=training_columns, fill_value=0)

    # Forecast the prices using the trained model
    forecasted_prices = model.predict(features)

    # Ensure future_dates do not extend beyond a reasonable range
    last_date = df['date'].max()
    num_predictions = min(len(forecasted_prices), max_periods)

    try:
        # Generate future dates with a maximum limit on the number of periods
        future_dates = pd.date_range(start=last_date, periods=num_predictions + 1, freq='M')[1:]
    except Exception as e:
        # Return an error message if date generation fails
        return None, f"Error generating future dates: {str(e)}"

    return future_dates, forecasted_prices[:num_predictions]






def handle_linear_regression(csv_file_path, independent_vars, dependent_var):
    # Load the dataset
    data = pd.read_csv(csv_file_path)

    # Check for missing values
    if data.isnull().values.any():
        print("Data contains missing values. Cleaning data...")

        # Option 1: Drop rows with missing values
        data = data.dropna(subset=independent_vars + [dependent_var])

    # Feature Selection
    X = data[independent_vars]  # Independent variables
    y = data[dependent_var]  # Dependent variable

    # Data Preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # Calculate and return results
    result = {
        'intercept': model.intercept_,
        'coefficients': dict(zip(independent_vars, model.coef_)),
        'predictions': predictions.tolist(),
        'actual_values': y_test.tolist()
    }

    return result


def handle_linear_regression_by_date(csv_file_path, independent_vars, dependent_var, start_date, end_date):
    # Load the dataset
    data = pd.read_csv(csv_file_path)

    # Ensure the 'date' column is in datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Filter the dataset based on the date range
    if start_date and end_date:
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    # Drop rows with null values in the independent and dependent variables
    data = data.dropna(subset=[dependent_var] + independent_vars)

    # Feature Selection
    X = data[independent_vars]  # Independent variables
    y = data[dependent_var]  # Dependent variable

    # Data Preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features (optional, depending on your data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # Calculate and return results
    result = {
        'intercept': model.intercept_,
        'coefficients': dict(zip(independent_vars, model.coef_)),
        'predictions': predictions.tolist(),
        'actual_values': y_test.tolist()
    }

    return result


def get_linear_regression(dataset, linearXaxis, linearYaxis):
    # Convert dataset to DataFrame
    df = pd.DataFrame(dataset)

    # Ensure linearXaxis and linearYaxis are lists
    if not isinstance(linearXaxis, list):
        linearXaxis = [linearXaxis]
    if not isinstance(linearYaxis, list):
        linearYaxis = [linearYaxis]

    # Validate that linearXaxis and linearYaxis are present in the dataset
    for col in linearXaxis:
        if col not in df.columns:
            return jsonify({'error': f"Column '{col}' not found in dataset."}), 400
    if linearYaxis[0] not in df.columns:
        return jsonify({'error': f"Column '{linearYaxis[0]}' not found in dataset."}), 400

    # Prepare data for regression
    X = df[linearXaxis]

    # Handle categorical target variables
    if df[linearYaxis[0]].dtype == 'object':
        encoder = LabelEncoder()
        df[linearYaxis[0]] = encoder.fit_transform(df[linearYaxis[0]])

    y = df[linearYaxis[0]]

    # Initialize and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    coefficients = model.coef_.tolist()
    intercept = model.intercept_

    # Prepare response
    response = {
        'actuals': y.tolist(),
        'predictions': predictions.tolist(),
        'mean_squared_error': mse,
        'r2_score': r2,
        'coefficients': coefficients,
        'intercept': intercept
    }

    return response

# def get_linear_regression(dataset, linearXaxis, linearYaxis):
#     # Convert dataset to DataFrame
#     df = pd.DataFrame(dataset)
#
#     # Ensure linearXaxis and linearYaxis are lists
#     if not isinstance(linearXaxis, list):
#         linearXaxis = [linearXaxis]
#     if not isinstance(linearYaxis, list):
#         linearYaxis = [linearYaxis]
#
#     # Validate that linearXaxis and linearYaxis are present in the dataset
#     for col in linearXaxis:
#         if col not in df.columns:
#             return jsonify({'error': f"Column '{col}' not found in dataset."}), 400
#     for col in linearYaxis:
#         if col not in df.columns:
#             return jsonify({'error': f"Column '{col}' not found in dataset."}), 400
#
#     # Prepare data for regression
#     X = df[linearXaxis]
#
#     # Initialize a dictionary to store results for each dependent variable
#     results = {}
#
#     for y_col in linearYaxis:
#         # Handle categorical target variables
#         if df[y_col].dtype == 'object':
#             encoder = LabelEncoder()
#             df[y_col] = encoder.fit_transform(df[y_col])
#
#         y = df[y_col]
#
#         # Initialize and fit the model
#         model = LinearRegression()
#         model.fit(X, y)
#
#         # Make predictions
#         predictions = model.predict(X)
#
#         # Calculate metrics
#         mse = mean_squared_error(y, predictions)
#         r2 = r2_score(y, predictions)
#         coefficients = model.coef_.tolist()
#         intercept = model.intercept_
#
#         # Store the results for the current dependent variable
#         results[y_col] = {
#             'actuals': y.tolist(),
#             'predictions': predictions.tolist(),
#             'mean_squared_error': mse,
#             'r2_score': r2,
#             'coefficients': coefficients,
#             'intercept': intercept
#         }
#
#     return results



# Example usage for handling linear regression and forecasting:
file_path = '/Users/ilmeedesilva/Downloads/wfp_food_prices_lka.csv'
independent_vars = ['feature1', 'feature2', 'feature3']
dependent_var = 'price'
