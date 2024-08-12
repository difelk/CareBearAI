import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
