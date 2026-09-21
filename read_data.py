def read_wq_taiwan(test_size=0.25, seed=42):

    from sklearn.model_selection import train_test_split
    from io import BytesIO
    from sklearn.preprocessing import MinMaxScaler

    import numpy as np
    import pandas as pd
    import requests

    key = '1a5DReajqstsnUSUdTcRm8pZqeIP9ZmOct834UcOLmjg'

    # Access the dataset through the Google Sheets link
    link = 'https://docs.google.com/spreadsheet/ccc?key=' + key + '&output=csv'
    r = requests.get(link)
    data = r.content

    # Read the CSV file with settings to avoid warnings
    df = pd.read_csv(BytesIO(data), header=0, low_memory=False)

    # Select and transform the relevant columns
    cols = ['siteid', 'sampledate', 'itemengabbreviation', 'itemvalue']
    data = df[cols]

    # Pivot the data
    data = data.pivot(
        index=['siteid', 'sampledate'],
        columns='itemengabbreviation',
        values='itemvalue'
    )

    # Add the site column
    data['site'] = [data.index[i][0] for i in range(len(data))]

    # Filter the data
    data = data[data['site'] < 1008]

    # Select the columns of interest
    cols = ['EC', 'RPI', 'SS', 'WT', 'pH']
    X = data[cols].copy()

    # Convert columns to numeric and handle invalid values
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')

    # Remove missing values
    X = X.dropna()

    # Define independent and dependent variables
    variable_names = ['EC', 'SS', 'WT', 'pH']
    target_names = ['RPI']

    # Normalize the independent variables
    # scaler = MinMaxScaler()
    # X[variable_names] = scaler.fit_transform(X[variable_names])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X[variable_names],
        X[target_names],
        test_size=test_size,
        random_state=seed
    )

    # Store information about the dataset
    n_samples, n_features = X_train.shape
    dataset = {
        'task': 'regression',
        'name': 'WQ Taiwan',
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train.values,
        'y_train': y_train.values.ravel(),
        'X_test': X_test.values,
        'y_test': y_test.values.ravel(),
        'targets': target_names,
        'descriptions': 'None',
        'reference': "https://data.moenv.gov.tw/en/dataset/detail/WQX_P_01",
    }

    return dataset

