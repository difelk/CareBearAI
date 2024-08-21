import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import pickle
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from scipy import stats
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_samples
from app.config import get_csv_file_path

# Suppress warnings
warnings.filterwarnings('ignore')


def km_load_data(file_path):
    return pd.read_csv(file_path)


def km_explore_data(df):
    return {
        "head": df.head().to_dict(orient='records'),
        "tail": df.tail().to_dict(orient='records'),
        "info": str(df.info()),  # Convert info output to string
        "description": df.describe().to_dict()
    }


def km_preprocess_data(data):
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

    return data


def km_split_data(data):
    return train_test_split(data, test_size=0.2, random_state=100)


def km_train_model(x_train, num_clusters=3):
    model = KMeans(n_clusters=num_clusters, random_state=100)
    model.fit(x_train)
    return model


def km_evaluate_model(model, x_test):
    # Predict clusters for the test data
    labels = model.predict(x_test)

    # Compute silhouette score
    score = silhouette_score(x_test, labels)
    sample_silhouette_values = silhouette_samples(x_test, labels)

    # Calculate cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(map(str, unique_labels), map(int, counts)))

    # Create a dictionary for clusters
    clusters = {str(label): np.where(labels == label)[0].tolist() for label in unique_labels}

    # Prepare feature values for each cluster using reconstructed 'commodity' and 'price'
    cluster_features = {}
    scatter_plot_data = {}

    if isinstance(x_test, pd.DataFrame):
        # Identify one-hot encoded columns for commodities
        commodity_columns = [col for col in x_test.columns if col.startswith('commodity_')]

        for label, indices in clusters.items():
            try:
                # Reconstruct commodities for each data point in the cluster
                reconstructed_commodities = [
                    [commodity.replace('commodity_', '') for commodity in commodity_columns if row[commodity] == 1]
                    for _, row in x_test.iloc[indices].iterrows()
                ]

                # Extract the price column
                prices = x_test.iloc[indices]['price'].values

                # Combine reconstructed commodities and prices
                cluster_features[label] = list(zip(reconstructed_commodities, prices))

                # Prepare data for scatter plot
                scatter_plot_data[label] = {
                    "x": [" & ".join(commodity_list) for commodity_list in reconstructed_commodities],
                    # combined 'commodity' labels
                    "y": prices.tolist()  # 'price'
                }
            except KeyError as e:
                print(f"KeyError while accessing DataFrame: {e}")
                print(f"Indices: {indices}")
    else:
        raise TypeError("x_test must be a DataFrame to access columns by name")

    # Sample silhouette values summary
    silhouette_values_summary = {
        "min": float(np.min(sample_silhouette_values)),
        "max": float(np.max(sample_silhouette_values)),
        "mean": float(np.mean(sample_silhouette_values)),
        "std": float(np.std(sample_silhouette_values))
    }

    # Interpretation
    interpretation = {
        "silhouette_score": float(score),
        "silhouette_score_interpretation": "",
        "cluster_sizes": cluster_sizes,
        "cluster_sizes_interpretation": "",
        "sample_silhouette_values_summary": silhouette_values_summary,
        "sample_silhouette_values_interpretation": "",
        "clusters": clusters,
        "cluster_features": cluster_features
    }

    # Silhouette score interpretation
    if score >= 0.7:
        interpretation[
            "silhouette_score_interpretation"] = "Clusters are very well-separated. The clustering is excellent."
    elif score >= 0.5:
        interpretation[
            "silhouette_score_interpretation"] = "Clusters are fairly well-separated. The clustering is good, but there might be room for improvement."
    elif score >= 0.3:
        interpretation[
            "silhouette_score_interpretation"] = "Clusters are somewhat separated, but the clustering quality is moderate. Consider re-evaluating the number of clusters or features."
    else:
        interpretation[
            "silhouette_score_interpretation"] = "Clusters are not well-separated. The clustering might need significant improvement or reconsideration of the clustering method."

    # Cluster sizes interpretation
    sizes = list(cluster_sizes.values())
    if all(size > 5 for size in sizes):
        interpretation[
            "cluster_sizes_interpretation"] = "Clusters have a sufficient number of points. The clustering is likely balanced and well-defined."
    elif any(size < 5 for size in sizes):
        interpretation[
            "cluster_sizes_interpretation"] = "Some clusters have very few points. This might indicate small or possibly irrelevant clusters that may need further examination."
    else:
        interpretation[
            "cluster_sizes_interpretation"] = "Cluster sizes vary significantly. This could suggest issues with cluster definition or the presence of outliers."

    # Sample silhouette values interpretation
    mean_silhouette = silhouette_values_summary["mean"]
    std_silhouette = silhouette_values_summary["std"]

    if mean_silhouette >= 0.5 and std_silhouette < 0.1:
        interpretation[
            "sample_silhouette_values_interpretation"] = "Most points have high silhouette values and there is low variability, indicating consistent and good clustering quality."
    elif mean_silhouette >= 0.3 and std_silhouette < 0.3:
        interpretation[
            "sample_silhouette_values_interpretation"] = "Clustering is moderate with some variability in silhouette values. This suggests overall reasonable clustering, but with some inconsistencies."
    else:
        interpretation[
            "sample_silhouette_values_interpretation"] = "Clustering has significant variability in silhouette values, indicating inconsistency in clustering quality. Further investigation or adjustments might be needed."

    return {
        "interpretation": interpretation,
        "scatter_plot_data": scatter_plot_data
    }


def km_forecast_clusters(data, model):
    result = model.predict(data)
    return result


def km_visualize_clusters(data, model):
    data_with_labels = data.copy()
    data_with_labels['cluster'] = model.predict(data)
    return data_with_labels


def handle_clustering(data_path, num_clusters, features):
    # Load and filter data
    data = pd.read_csv(data_path)
    filtered_data = data[features].dropna()  # Drop NaN values for simplicity

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(filtered_data)

    # Train K-Means model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    filtered_data['cluster'] = kmeans.fit_predict(scaled_features)

    # Prepare results
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    result = {
        'clustered_data': filtered_data.to_dict(orient='records'),
        'cluster_centers': cluster_centers.tolist()
    }

    return result


def km_cluster_insights(data_with_labels):
    insights = {}
    grouped = data_with_labels.groupby('cluster')

    # Debugging: Print the columns of the first group to ensure 'market' and 'commodity' are present
    for cluster, group in grouped:
        print(f"Cluster {cluster} columns:", group.columns.tolist())
        break

    for cluster, group in grouped:
        # Assuming that 'commodity' and 'market' columns have been transformed to one-hot encoded columns
        commodity_columns = [col for col in group.columns if col.startswith('commodity_')]
        market_columns = [col for col in group.columns if col.startswith('market_')]

        # Reconstruct commodity and market lists from one-hot encoded columns
        commodities = [col.replace('commodity_', '') for col in commodity_columns if group[col].any()]
        markets = [col.replace('market_', '') for col in market_columns if group[col].any()]

        insights[cluster] = {
            'mean_price': group['price'].mean(),
            'median_price': group['price'].median(),
            'price_range': (group['price'].min(), group['price'].max()),
            'markets': markets,
            'commodities': commodities
        }

    return insights


def interpret_forecasted_clusters(forecasted_clusters):
    from collections import Counter

    # Convert to a set to find unique clusters
    unique_clusters = set(forecasted_clusters)

    # Initialize the interpretation dictionary
    interpretation = {}

    if len(unique_clusters) == 1:
        # All points are assigned to the same cluster
        interpretation['summary'] = "All data points are forecasted to belong to a single cluster."
        interpretation['details'] = {
            'cluster': list(unique_clusters)[0],
            'description': "All data points share similar characteristics and are grouped into one cluster."
        }
    else:
        # Check distribution of clusters
        cluster_counts = Counter(forecasted_clusters)
        total_points = len(forecasted_clusters)
        most_common_cluster, most_common_count = cluster_counts.most_common(1)[0]

        if len(cluster_counts) == 1:
            interpretation['summary'] = "All data points are assigned to the same cluster."
            interpretation['details'] = {
                'cluster': list(unique_clusters)[0],
                'description': "All data points share similar characteristics and are grouped into one cluster."
            }
        elif len(cluster_counts) == len(forecasted_clusters):
            interpretation['summary'] = "Each data point is assigned to a unique cluster."
            interpretation['details'] = "Each data point is isolated in its own cluster, indicating high variability " \
                                        "among the data points."
        elif most_common_count / total_points > 0.5:
            interpretation['summary'] = (f"The majority of data points are assigned to cluster {most_common_cluster}.")
            interpretation['details'] = {
                'cluster': most_common_cluster,
                'description': (f"A significant portion of data points belongs to cluster {most_common_cluster}. "
                                "This indicates that there is a predominant cluster with similar characteristics.")
            }
        else:
            interpretation['summary'] = "Clusters are distributed across the data points with no dominant cluster."
            interpretation[
                'details'] = "The data points are spread across multiple clusters without a clear dominant group."

    return interpretation


# Example usage
file_path = get_csv_file_path()
data = km_load_data(file_path)
explored_data = km_explore_data(data)
preprocessed_data = km_preprocess_data(data)
x_train, x_test = km_split_data(preprocessed_data)
model = km_train_model(x_train)
evaluation_results = km_evaluate_model(model, x_test)
forecasted_clusters = km_forecast_clusters(x_test, model)
visualized_data = km_visualize_clusters(preprocessed_data, model)
cluster_insights = km_cluster_insights(visualized_data)

print("Evaluation Results:", evaluation_results)
print("Forecasted Clusters:", forecasted_clusters)
print("Cluster Insights:", json.dumps(cluster_insights, indent=2))
