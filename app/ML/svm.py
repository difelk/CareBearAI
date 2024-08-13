import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error,
                             r2_score, roc_curve, auc, precision_recall_curve)
import io
import base64
from scipy import stats

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

    median_price = data['price'].median()
    data['price_class'] = (data['price'] > median_price).astype(int)
    target = data['price_class']
    features = data.drop(columns=['price', 'price_class'])

    return features, target


def svm_split_data(features, target):
    return train_test_split(features, target, train_size=0.8, test_size=0.2, random_state=100)


def svm_train_model(x_train, y_train):
    svc = SVC()
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1,
                               error_score='raise')
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_, grid_search


def svm_evaluate_model(file_path):
    data = svm_load_data(file_path)
    features, target = svm_preprocess_data(data)
    x_train, x_test, y_train, y_test = svm_split_data(features, target)
    best_svc, grid_search = svm_train_model(x_train, y_train)

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