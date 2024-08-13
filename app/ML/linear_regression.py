import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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
