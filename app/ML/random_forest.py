import warnings
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, \
    mean_squared_error, r2_score
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    return pd.read_csv(file_path)


def explore_data(df):
    return {
        "head": df.head().to_dict(orient='records'),
        "tail": df.tail().to_dict(orient='records'),
        "info": str(df.info()),  # Convert info output to string
        "description": df.describe().to_dict()
    }


def preprocess_data(data):
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
    for column in categorical_columns:
        if column in data.columns:
            data = pd.get_dummies(data, columns=[column])

    scaler = MinMaxScaler()
    numerical_features = ['latitude', 'longitude', 'price', 'usdprice', 'USD RATE']
    for feature in numerical_features:
        if feature in data.columns:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
    data = data.dropna()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    median_price = data['price'].median()
    data['price_class'] = (data['price'] > median_price).astype(int)
    target = data['price_class']
    features = data.drop(columns=['price', 'price_class'])

    return features, target


def split_data(features, target):
    return train_test_split(features, target, train_size=0.8, test_size=0.2, random_state=100)


def train_model(x_train, y_train):
    rf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1,
                               error_score='raise')
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_, grid_search


def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)


def evaluate_model(df):
    # Preprocess the data
    features, target = preprocess_data(df)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(features, target, train_size=0.8, test_size=0.2,
                                                        random_state=100)

    # Train the model
    best_rf, grid_search = train_model(x_train, y_train)

    # Predictions
    y_pred = best_rf.predict(x_test)

    # Evaluation metrics
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "cv_scores": cross_val_score(best_rf, x_test, y_test, cv=5).tolist(),
        "grid_search_results": pd.DataFrame(grid_search.cv_results_).sort_values(by='mean_test_score',
                                                                                 ascending=False).to_dict(
            orient='records')
    }



def get_feature_importances(best_rf, features):
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10
    top_indices = indices[:top_n]
    return {
        "top_features": features.columns[top_indices].tolist(),
        "top_importances": importances[top_indices].tolist()
    }


def create_plots(features, target, best_rf, evaluation_results):
    plot_paths = {}

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

    # Plot: Top Feature Importances
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_feature_names = features.columns[top_indices]

    plt.figure(figsize=(14, 8))
    plt.title("Top Feature Importances", fontsize=16)
    plt.bar(range(top_n), top_importances, align="center")
    plt.xticks(range(top_n), top_feature_names, rotation=90, fontsize=8)
    plt.xlim([-1, top_n])
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.tight_layout()
    top_feature_importances_path = '/Users/ilmeedesilva/Desktop/ML Ass 4/top_feature_importances.png'
    plt.savefig(top_feature_importances_path)
    plt.close()
    plot_paths['top_feature_importances'] = top_feature_importances_path

    # Plot: Actual vs. Predicted Prices
    y_test = evaluation_results.get('y_test')
    y_pred = evaluation_results.get('y_pred')

    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolor='k')
    plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], 'k--', lw=2)
    plt.xlabel('Actual Prices', fontsize=14)
    plt.ylabel('Predicted Prices', fontsize=14)
    plt.title('Actual vs Predicted Prices', fontsize=16)
    plt.tight_layout()
    actual_vs_predicted_path = '/Users/ilmeedesilva/Desktop/ML Ass 4/actual_vs_predicted.png'
    plt.savefig(actual_vs_predicted_path)
    plt.close()
    plot_paths['actual_vs_predicted'] = actual_vs_predicted_path

    # Plot: Residuals vs Predicted Prices
    residuals = np.array(y_test) - np.array(y_pred)

    plt.figure(figsize=(12, 8))
    plt.scatter(y_pred, residuals, alpha=0.5, edgecolor='k')
    plt.hlines(0, xmin=min(y_pred), xmax=max(y_pred), colors='r', linestyles='--')
    plt.xlabel('Predicted Prices', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.title('Residuals vs Predicted Prices', fontsize=16)
    plt.tight_layout()
    residuals_vs_predicted_path = '/Users/ilmeedesilva/Desktop/ML Ass 4/residuals_vs_predicted.png'
    plt.savefig(residuals_vs_predicted_path)
    plt.close()
    plot_paths['residuals_vs_predicted'] = residuals_vs_predicted_path

    # Plot: Residual Histogram
    plt.figure(figsize=(12, 8))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('Residual', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Residuals', fontsize=16)
    plt.tight_layout()
    residual_histogram_path = '/Users/ilmeedesilva/Desktop/ML Ass 4/residual_histogram.png'
    plt.savefig(residual_histogram_path)
    plt.close()
    plot_paths['residual_histogram'] = residual_histogram_path

    # Plot: Cross-Validation Score Distribution
    cv_scores = evaluation_results.get('cv_scores')

    plt.figure(figsize=(12, 8))
    plt.boxplot(cv_scores, vert=False)
    plt.xlabel('Cross-Validation Score', fontsize=14)
    plt.title('Cross-Validation Score Distribution', fontsize=16)
    plt.tight_layout()
    cv_score_distribution_path = '/Users/ilmeedesilva/Desktop/ML Ass 4/cv_score_distribution.png'
    plt.savefig(cv_score_distribution_path)
    plt.close()
    plot_paths['cv_score_distribution'] = cv_score_distribution_path

    # Plot: Grid Search Results
    results = evaluation_results.get('grid_search_results')

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=results, x='param_n_estimators', y='mean_test_score', hue='param_max_depth', marker='o')
    plt.xlabel('Number of Estimators', fontsize=14)
    plt.ylabel('Mean Test Score', fontsize=14)
    plt.title('Grid Search Results', fontsize=16)
    plt.legend(title='Max Depth', title_fontsize='13', fontsize='12')
    plt.tight_layout()
    grid_search_results_path = '/Users/ilmeedesilva/Desktop/ML Ass 4/grid_search_results.png'
    plt.savefig(grid_search_results_path)
    plt.close()
    plot_paths['grid_search_results'] = grid_search_results_path

    # Plot: Filtered Feature Correlation Heatmap
    threshold = 0.5
    corr_matrix = features.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    feature_correlation_heatmap_path = '/Users/ilmeedesilva/Desktop/ML Ass 4/feature_correlation_heatmap.png'
    plt.savefig(feature_correlation_heatmap_path)
    plt.close()
    plot_paths['feature_correlation_heatmap'] = feature_correlation_heatmap_path

    return plot_paths


def main():
    # Load data
    file_path = '/Users/ilmeedesilva/Downloads/wfp_food_prices_lka.csv'
    data = load_data(file_path)

    # Data exploration
    exploration_results = explore_data(data)
    print(exploration_results)

    # Data preprocessing
    features, target = preprocess_data(data)

    # Split data
    x_train, x_test, y_train, y_test = split_data(features, target)

    # Train model
    best_rf, grid_search = train_model(x_train, y_train)

    # Evaluate model
    evaluation_results = evaluate_model(data)

    # Feature importances
    feature_importances = get_feature_importances(best_rf, features)
    print(feature_importances)

    # Create plots
    plot_paths = create_plots(features, target, best_rf, evaluation_results)
    print(plot_paths)

    return {
        "exploration_results": exploration_results,
        "evaluation_results": evaluation_results,
        "plot_paths": plot_paths
    }


if __name__ == "__main__":
    main()
