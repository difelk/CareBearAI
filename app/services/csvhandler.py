import pandas as pd


def get_all_csv_data(csv_file):
    data = pd.read_csv(csv_file)
    return data.to_dict(orient='records')


def extract_header(csv_file):
    data = pd.read_csv(csv_file)
    headers = data.columns.tolist()
    return headers
