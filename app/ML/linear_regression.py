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

warnings.filterwarnings('ignore')


def lr_load_data(file_path):
    return pd.read_csv(file_path)


def lr_explore_data(df):
    return {
        "head": df.head().to_dict(orient='records'),
        "tail": df.tail().to_dict(orient='records'),
        "info": str(df.info()),
        "description": df.describe().to_dict()
    }


def lr_preprocess_data(data, training_features=None):
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
    outliers = (z_scores > 3).any(axis=1)
    data = data[~outliers]
    data = data.dropna()

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        data = data.drop(columns=['date'])

    categorical_columns = ['admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'currency', 'priceflag',
                           'pricetype']
    data = pd.get_dummies(data, columns=[col for col in categorical_columns if col in data.columns])

    scaler = MinMaxScaler()
    numerical_features = ['latitude', 'longitude', 'price', 'usdprice', 'USD RATE']
    for feature in numerical_features:
        if feature in data.columns:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
            data[feature] = scaler.fit_transform(data[[feature]])

    if training_features:

        for col in training_features:
            if col not in data.columns:
                data[col] = 0

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
    features, target = lr_preprocess_data(df)

    x_train, x_test, y_train, y_test = train_test_split(features, target, train_size=0.8, test_size=0.2,
                                                        random_state=100)

    model = lr_train_model(x_train, y_train)

    y_pred = model.predict(x_test)

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
    filtered_df = filter_data(df, market, category, commodity)

    features, _ = lr_preprocess_data(filtered_df)

    training_columns = model.feature_names_in_
    features = features.reindex(columns=training_columns, fill_value=0)

    forecasted_prices = model.predict(features)

    last_date = df['date'].max()
    num_predictions = min(len(forecasted_prices), max_periods)

    try:

        future_dates = pd.date_range(start=last_date, periods=num_predictions + 1, freq='M')[1:]
    except Exception as e:

        return None, f"Error generating future dates: {str(e)}"

    return future_dates, forecasted_prices[:num_predictions]


def handle_linear_regression(csv_file_path, independent_vars, dependent_var):
    data = pd.read_csv(csv_file_path)

    if data.isnull().values.any():
        print("Data contains missing values. Cleaning data...")

        data = data.dropna(subset=independent_vars + [dependent_var])

    X = data[independent_vars]
    y = data[dependent_var]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    result = {
        'intercept': model.intercept_,
        'coefficients': dict(zip(independent_vars, model.coef_)),
        'predictions': predictions.tolist(),
        'actual_values': y_test.tolist()
    }

    return result


def handle_linear_regression_by_date(csv_file_path, independent_vars, dependent_var, start_date, end_date):
    data = pd.read_csv(csv_file_path)

    data['date'] = pd.to_datetime(data['date'])

    if start_date and end_date:
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    data = data.dropna(subset=[dependent_var] + independent_vars)

    X = data[independent_vars]
    y = data[dependent_var]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    # Check if columns exist in the dataset
    for col in linearXaxis:
        if col not in df.columns:
            return jsonify({'error': f"Column '{col}' not found in dataset."}), 400
    if linearYaxis[0] not in df.columns:
        return jsonify({'error': f"Column '{linearYaxis[0]}' not found in dataset."}), 400

    # Data Cleaning

    # missing values removing
    df = df.dropna()

    # duplicates
    df = df.drop_duplicates()

    #indip columns to numeric if needed
    for col in linearXaxis:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. non numeric Y
    if df[linearYaxis[0]].dtype == 'object':
        encoder = LabelEncoder()
        df[linearYaxis[0]] = encoder.fit_transform(df[linearYaxis[0]])

    X = df[linearXaxis].values
    y = df[linearYaxis[0]].values


    model = LinearRegression()
    model.fit(X, y)


    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    coefficients = model.coef_.tolist()
    intercept = model.intercept_

    response = {
        'actuals': y.tolist(),
        'predictions': predictions.tolist(),
        'mean_squared_error': mse,
        'r2_score': r2,
        'coefficients': coefficients,
        'intercept': intercept
    }

    return response


file_path = '/Users/ilmeedesilva/Downloads/wfp_food_prices_lka.csv'
independent_vars = ['feature1', 'feature2', 'feature3']
dependent_var = 'price'
