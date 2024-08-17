import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


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
