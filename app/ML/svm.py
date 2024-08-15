import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error,
                             r2_score, roc_curve, auc, precision_recall_curve)
import io
import base64
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

from sklearn.decomposition import PCA

# Suppress warnings
warnings.filterwarnings('ignore')


def svm_load_data(file_path):
    return pd.read_csv(file_path)


def svm_explore_data(df):
    return {
        "head": df.head().to_dict(orient='records'),
        "tail": df.tail().to_dict(orient='records'),
        "info": str(df.info()),  # Convert info output to string
        "description": df.describe().to_dict()
    }


def svm_preprocess_data(data):
    # Identify and Handle Outliers
    # Use Z-score for identifying outliers
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
    outliers = (z_scores > 3).any(axis=1)
    data_no_outliers = data[~outliers]
    print(f"Number of outliers removed: {sum(outliers)}")

    data = data_no_outliers  # Use the data without outliers

    data = data.dropna()  # Dropping rows with missing values

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        data = data.drop(columns=['date'])

    categorical_columns = ['admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'currency', 'priceflag',
                           'pricetype']
    data = pd.get_dummies(data, columns=categorical_columns)

    scaler = StandardScaler()
    numerical_features = ['latitude', 'longitude', 'price', 'usdprice', 'USD RATE']
    for feature in numerical_features:
        if feature in data.columns:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
    data = data.dropna()  # Drop rows with NaN values created by coercion
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Classify prices into "high" and "low" based on the median price
    median_price = data['price'].median()
    data['price_class'] = (data['price'] > median_price).astype(int)  # 0 for "low", 1 for "high"
    target = data['price_class']
    features = data.drop(columns=['price', 'price_class'])

    return features, target, scaler


def svm_split_data(features, target):
    return train_test_split(features, target, train_size=0.8, test_size=0.2, random_state=100)


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV


def svm_train_model(x_train, y_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    svc = SVC()
    param_dist = {
        'C': [0.1, 1],
        'kernel': ['linear'],
        'gamma': ['scale']
    }

    random_search = RandomizedSearchCV(estimator=svc, param_distributions=param_dist,
                                       n_iter=5,
                                       cv=3,
                                       scoring='accuracy',
                                       n_jobs=-1, random_state=42, error_score='raise')

    random_search.fit(x_train_scaled, y_train)

    # Capture training feature columns and data types
    feature_names = x_train.columns.tolist()
    training_dtypes = x_train.dtypes

    return random_search.best_estimator_, random_search, scaler, feature_names, training_dtypes


def svm_evaluate_model(file_path):
    data = svm_load_data(file_path)
    features, target, scaler = svm_preprocess_data(data)
    x_train, x_test, y_train, y_test = svm_split_data(features, target)
    best_svc, grid_search, scaler, feature_columns, training_dtypes = svm_train_model(x_train,
                                                                                      y_train)

    # Ensure test data has the same columns as the training data
    x_test = x_test.reindex(columns=feature_columns, fill_value=0)

    y_pred = best_svc.predict(x_test)
    y_prob = best_svc.decision_function(x_test)

    accuracy = np.mean(y_pred == y_test)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(best_svc, features, target, cv=5).tolist()
    grid_search_results = pd.DataFrame(grid_search.cv_results_).to_dict(orient='records')

    return {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "r2_score": r2,
        "cv_scores": cv_scores,
        "grid_search_results": grid_search_results,
        "roc_curve": svm_generate_roc_curve(y_test, y_prob),
        "precision_recall_curve": svm_generate_precision_recall_curve(y_test, y_prob)
    }


def svm_generate_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": roc_auc
    }


def svm_generate_precision_recall_curve(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "pr_auc": pr_auc
    }


def svm_forecast_prices(data, model, commodity=None, market=None, category=None):
    # Filter data for the specified commodity, market, and category
    if commodity:
        data = data[data['commodity'] == commodity]
    if market:
        data = data[data['market'] == market]
    if category:
        data = data[data['category'] == category]

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')

    # Resample to monthly data and interpolate missing values
    if 'price' not in data.columns or data.empty:
        raise ValueError("No 'price' column or empty data after filtering.")

    price_data = data['price'].resample('M').mean().interpolate()

    if price_data.empty:
        raise ValueError("No data available for forecasting after resampling.")

    # Fit ARIMA model
    arima_model = ARIMA(price_data, order=(1, 1, 1))  # Adjust order as needed
    model_fit = arima_model.fit()

    # Forecast for the next 12 months
    forecast = model_fit.forecast(steps=12)

    return forecast


def svm_forecast_all_commodities(data, model):
    commodities = data['commodity'].unique()
    forecasts = {}

    for commodity in commodities:
        forecasts[commodity] = svm_forecast_prices(data, model, commodity=commodity).tolist()

    return forecasts


def svm_forecast_all_markets(data, model):
    markets = data['market'].unique()
    forecasts = {}

    for market in markets:
        forecasts[market] = svm_forecast_prices(data, model, market=market).tolist()

    return forecasts


def svm_forecast_all_categories(data, model):
    categories = data['category'].unique()
    forecasts = {}

    for category in categories:
        forecasts[category] = svm_forecast_prices(data, model, category=category).tolist()

    return forecasts


def get_historical_averages(data, end_date, months=6):
    start_date = end_date - timedelta(days=30 * months)
    historical_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    # Compute average prices for each commodity
    historical_averages = historical_data.groupby('commodity').agg({
        'price': 'mean',
        'USD RATE': 'mean'
    }).reset_index()
    historical_averages.rename(columns={'price': 'avg_price', 'USD RATE': 'avg_usd_rate'}, inplace=True)

    return historical_averages


def prepare_forecast_data(data, historical_averages):
    # Merge historical averages with the new data
    forecast_data = data.merge(historical_averages, on='commodity', how='left')

    # Fill NaN values if there are commodities with no historical data
    forecast_data['avg_price'].fillna(forecast_data['price'].mean(), inplace=True)
    forecast_data['avg_usd_rate'].fillna(forecast_data['USD RATE'].mean(), inplace=True)

    return forecast_data


# Forecast price classification (high or low)
from statsmodels.tsa.arima.model import ARIMA


# def svm_forecast_price_class(model, new_data, feature_names, training_dtypes, start_date, end_date, scaler=None):
#     # Process date if it exists in the new_data
#     if 'date' in new_data.columns:
#         new_data['date'] = pd.to_datetime(new_data['date'])
#         new_data = new_data[(new_data['date'] >= start_date) & (new_data['date'] <= end_date)]
#         new_data['year'] = new_data['date'].dt.year
#         new_data['month'] = new_data['date'].dt.month
#         new_data['day'] = new_data['date'].dt.day
#         new_data = new_data.drop(columns=['date'])
#
#     # Convert categorical columns to dummy variables
#     categorical_columns = ['admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'currency', 'priceflag',
#                            'pricetype']
#     new_data = pd.get_dummies(new_data, columns=categorical_columns, drop_first=False)
#
#     # Add missing columns with zeros
#     missing_cols = set(feature_names) - set(new_data.columns)
#     for col in missing_cols:
#         new_data[col] = 0
#
#     # Remove columns that are not in the training feature names
#     new_data = new_data[[col for col in new_data.columns if col in feature_names]]
#
#     # Reorder columns to match training feature names
#     new_data = new_data[feature_names]
#
#     # Ensure data types match those used in training
#     for col in new_data.columns:
#         if col in training_dtypes:
#             dtype = training_dtypes[col]
#             if dtype == 'float64' or dtype == 'int64':
#                 new_data[col] = new_data[col].fillna(0).astype(dtype)
#             elif dtype == 'bool':
#                 new_data[col] = new_data[col].fillna(False).astype(dtype)
#             elif dtype == 'object':
#                 new_data[col] = new_data[col].fillna('Unknown').astype(dtype)
#             elif dtype == 'datetime64[ns]':
#                 default_date = pd.Timestamp('1900-01-01')
#                 new_data[col] = new_data[col].fillna(default_date).astype(dtype)
#             else:
#                 pass
#
#     # Convert DataFrame to NumPy array for prediction
#     new_data_array = new_data.values
#
#     # Predict and map the numeric predictions to labels
#     predictions = model.predict(new_data_array)
#     prediction_labels = ["low" if p == 0 else "high" for p in predictions]
#
#     # Extract commodity names from columns
#     commodity_columns = [col for col in feature_names if col.startswith('commodity_')]
#
#     result = new_data.copy()
#     result['price_class'] = prediction_labels
#
#     commodity_results = {}
#     for commodity in commodity_columns:
#         is_commodity_column = result[commodity] == 1
#         if is_commodity_column.any():
#             commodity_name = commodity.replace('commodity_', '')
#             commodity_prices = result.loc[is_commodity_column, 'avg_price']
#
#             if not commodity_prices.empty:
#                 avg_price = commodity_prices.mean()
#
#                 # Fit ARIMA model for the given commodity's average prices
#                 arima_model = ARIMA(commodity_prices, order=(1, 1, 1))  # Adjust order as needed
#                 model_fit = arima_model.fit()
#
#                 # Forecast for the next 1 month
#                 forecast_prices = model_fit.forecast(steps=1)
#
#                 # Determine high or low based on comparison with the avg price
#                 forecasted_class = ["high" if price > avg_price else "low" for price in forecast_prices]
#
#                 commodity_results[commodity_name] = {
#                     "forecasted_prices": forecast_prices.tolist(),
#                     "forecasted_class": forecasted_class,
#                     "average_price": avg_price
#                 }
#
#     return commodity_results

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def svm_forecast_price_class(model, new_data, feature_names, training_dtypes, scaler=None):
    # Process date if it exists in the new_data
    if 'date' in new_data.columns:
        new_data['date'] = pd.to_datetime(new_data['date'])
        new_data['year'] = new_data['date'].dt.year
        new_data['month'] = new_data['date'].dt.month
        new_data['day'] = new_data['date'].dt.day
        new_data = new_data.drop(columns=['date'])

    # Convert categorical columns to dummy variables
    categorical_columns = ['admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'currency', 'priceflag',
                           'pricetype']
    new_data = pd.get_dummies(new_data, columns=categorical_columns, drop_first=False)

    # Add missing columns with zeros
    missing_cols = set(feature_names) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0

    # Remove columns that are not in the training feature names
    new_data = new_data[[col for col in new_data.columns if col in feature_names]]

    # Reorder columns to match training feature names
    new_data = new_data[feature_names]

    # Ensure data types match those used in training
    for col in new_data.columns:
        if col in training_dtypes:
            dtype = training_dtypes[col]
            if dtype == 'float64' or dtype == 'int64':
                new_data[col] = new_data[col].fillna(0).astype(dtype)
            elif dtype == 'bool':
                new_data[col] = new_data[col].fillna(False).astype(dtype)
            elif dtype == 'object':
                new_data[col] = new_data[col].fillna('Unknown').astype(dtype)
            elif dtype == 'datetime64[ns]':
                default_date = pd.Timestamp('1900-01-01')
                new_data[col] = new_data[col].fillna(default_date).astype(dtype)
            else:
                pass

    # Debugging: Print feature columns and data types
    print("Training feature columns:", feature_names)
    print("\nNew data feature columns:", new_data.columns.tolist())
    print("\nNew data types:", new_data.dtypes)
    print("\nTraining data types:", training_dtypes)

    # Convert DataFrame to NumPy array for prediction
    new_data_array = new_data.values

    # Debugging: Print the shape of the new data array
    print("\nNew data shape:", new_data_array.shape)

    # Predict and map the numeric predictions to labels
    predictions = model.predict(new_data_array)
    prediction_labels = ["low" if p == 0 else "high" for p in predictions]

    # Extract commodity names from columns
    commodity_columns = [col for col in feature_names if col.startswith('commodity_')]

    result = new_data.copy()
    result['price_class'] = prediction_labels

    # Handle overall average prices for one-hot encoded commodities
    commodity_prefix = 'commodity_'
    commodities = [col.replace(commodity_prefix, '') for col in commodity_columns]

    overall_avg_prices = {}
    for commodity in commodities:
        commodity_data = new_data.filter(like=f'{commodity_prefix}{commodity}', axis=1)
        if not commodity_data.empty:
            avg_price = new_data.loc[commodity_data.any(axis=1), 'avg_price'].mean()
            overall_avg_prices[commodity] = avg_price

    commodity_results = {}
    for commodity in commodity_columns:
        commodity_name = commodity.replace('commodity_', '')

        # Check if the commodity exists in overall average prices
        if commodity_name not in overall_avg_prices:
            print(f"Warning: Commodity '{commodity_name}' not found in overall average prices.")
            continue

        is_commodity_column = result[commodity] == 1
        if is_commodity_column.any():
            # Calculate the average price for the given time period
            commodity_prices = result.loc[is_commodity_column, 'avg_price']
            avg_price_given_period = commodity_prices.mean()

            overall_avg_price = overall_avg_prices[commodity_name]

            # Fit ARIMA model for the given commodity's average prices
            arima_model = ARIMA(commodity_prices, order=(1, 1, 1))  # Adjust order as needed
            model_fit = arima_model.fit()

            # Forecast for the next 1 month
            forecast_prices = model_fit.forecast(steps=1)

            # Determine high or low based on comparison with the average price
            forecasted_class = ["high" if price > overall_avg_price else "low" for price in forecast_prices]

            commodity_results[commodity_name] = {
                "forecasted_prices": forecast_prices.tolist(),
                "forecasted_class": forecasted_class,
                "overall_avg_price": overall_avg_price
            }

    return commodity_results


def svm_create_plots(features, target, best_svc, evaluation_results):
    plot_paths = {}

    def save_plot(fig, title):
        img = io.BytesIO()
        fig.savefig(img, format='png')
        plt.close(fig)
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf-8')

        # Outlier Identification for Plotting

    numerical_data = features.select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numerical_data))
    outliers = (z_scores > 3).any(axis=1)
    features_with_outliers = features.copy()
    features_with_outliers['outlier'] = outliers

    # Plot: Outliers
    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(features_with_outliers['latitude'], features_with_outliers['longitude'],
                          c=features_with_outliers['outlier'], cmap='coolwarm', alpha=0.7)
    plt.xlabel('Latitude', fontsize=14)
    plt.ylabel('Longitude', fontsize=14)
    plt.title('Outliers Visualization', fontsize=16)
    plt.colorbar(scatter, label='Outlier')
    plt.tight_layout()
    outliers_plot_path = '/Users/ilmeedesilva/Desktop/ML Ass 4/outliers_plot.png'
    plt.savefig(outliers_plot_path)
    plt.close()
    plot_paths['outliers_plot'] = outliers_plot_path

    # Plot: Confusion Matrix Heatmap
    conf_matrix = np.array(evaluation_results['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix Heatmap', fontsize=16)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plot_paths['confusion_matrix'] = save_plot(fig, 'Confusion Matrix Heatmap')

    # Plot: Cross-Validation Score Distribution
    cv_scores = evaluation_results['cv_scores']
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.boxplot(cv_scores, vert=False)
    ax.set_xlabel('Cross-Validation Score', fontsize=14)
    ax.set_title('Cross-Validation Score Distribution', fontsize=16)
    plot_paths['cv_score_distribution'] = save_plot(fig, 'Cross-Validation Score Distribution')

    # Plot: Grid Search Results
    results = pd.DataFrame(evaluation_results['grid_search_results'])
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(data=results, x='param_C', y='mean_test_score', hue='param_kernel', marker='o', ax=ax)
    ax.set_xlabel('Regularization Parameter C', fontsize=14)
    ax.set_ylabel('Mean Test Score', fontsize=14)
    ax.set_title('Grid Search Results', fontsize=16)
    ax.legend(title='Kernel', title_fontsize='13', fontsize='12')
    plot_paths['grid_search_results'] = save_plot(fig, 'Grid Search Results')

    # Plot: Feature Importance (for Linear Kernel)
    if best_svc.kernel == 'linear':
        importances = best_svc.coef_[0]
        top_features = np.argsort(np.abs(importances))[-5:]
        top_feature_names = features.columns[top_features]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_feature_names, importances[top_features])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 5 Features by Importance', fontsize=16)
        plot_paths['feature_importance'] = save_plot(fig, 'Feature Importance')

    # Plot: Pairplot of Top Correlated Features
    correlations = features.corrwith(target)
    top_corr_features = correlations.abs().sort_values(ascending=False).head(5).index
    features_with_target = pd.concat([features[top_corr_features], target.reset_index(drop=True)], axis=1)
    features_with_target.columns = list(top_corr_features) + ['Price']
    fig = plt.figure(figsize=(12, 8))
    sns.pairplot(features_with_target, hue='Price', palette='viridis')
    plt.suptitle('Pairplot of Top Correlated Features', y=1.02, fontsize=16)
    plot_paths['pairplot'] = save_plot(fig, 'Pairplot of Top Correlated Features')

    # Plot: Feature Distribution (Histograms for each feature)
    features_hist = features.copy()
    features_hist['Price'] = target
    melted = features_hist.melt(id_vars='Price', var_name='Feature', value_name='Value')
    fig = plt.figure(figsize=(12, 8))
    g = sns.FacetGrid(melted, col='Feature', col_wrap=4, height=4)
    g.map_dataframe(sns.histplot, x='Value')
    g.set_axis_labels('Value', 'Frequency')
    g.set_titles(col_template="{col_name}")
    g.add_legend()
    plt.suptitle('Feature Distribution', y=1.02, fontsize=16)
    plot_paths['feature_distribution'] = save_plot(fig, 'Feature Distribution')

    return plot_paths
