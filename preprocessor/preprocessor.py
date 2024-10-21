import pandas as pd
from imblearn.over_sampling import SMOTE


def preprocess(file_path, n=200000):
    # Load the data, and only sample 'n' rows
    df = pd.read_csv(file_path)
    df = df.sample(n=n, random_state=1)  # Sample 'n' rows from the data
    df = df.drop_duplicates(keep='first')  # Remove duplicates

    # Split features and target
    X = df.drop(columns=['Is_laundering'])
    y = df['Is_laundering']  # Target

    # Convert time-related features
    X['Time'] = pd.to_datetime(X['Time'], format='%H:%M:%S')
    X['Date'] = pd.to_datetime(X['Date'], format='%Y-%m-%d')
    X['Year'] = X['Date'].dt.year
    X['Month'] = X['Date'].dt.month
    X['Day'] = X['Date'].dt.day
    X['Hour'] = X['Time'].dt.hour
    X['Minute'] = X['Time'].dt.minute
    X['Second'] = X['Time'].dt.second
    X = X.drop(columns=['Date', 'Time'])  # Drop original date and time columns

    # Convert categorical columns to one-hot encoding
    categorical_columns = [
        'Payment_currency',
        'Received_currency',
        'Sender_bank_location',
        'Receiver_bank_location',
        'Payment_type',
        'Laundering_type'
    ]
    X = pd.get_dummies(X, columns=categorical_columns)

    # Apply SMOTE for resampling
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)  # 对y也resample, 这样做对吗?

    # Create a new DataFrame from the resampled data
    data = pd.DataFrame(X_resampled, columns=X.columns)
    data['Is_laundering'] = y_resampled

    # Add high-risk country feature
    high_risk_countries = {"Nigeria", "Morocco", "Turkey", "Pakistan", "Mexico", "Spain"}
    sender_columns = [f"Sender_bank_location_{country}" for country in high_risk_countries]
    receiver_columns = [f"Receiver_bank_location_{country}" for country in high_risk_countries]

    # Check if any of the sender or receiver columns indicate a high-risk country
    data['high_risk_countries'] = data[sender_columns + receiver_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    # Convert boolean columns to integers
    bool_columns = data.select_dtypes(include=['bool', 'object']).columns
    for col in bool_columns:
        # Convert boolean-like object columns to actual boolean type first
        if data[col].dtype == 'object':
            data[col] = data[col].map({'True': True, 'False': False, 'true': True, 'false': False})
        data[col] = data[col].astype(int)

    # Handle any missing values in the target column
    if data['Is_laundering'].isna().sum() > 0:
        data = data.dropna(subset=['Is_laundering'])

    return data

# import pandas as pd
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import OneHotEncoder
#
# df = pd.read_csv("/content/SAML-D.csv")
# df = df.sample(n=200000, random_state=1)
# df = df.drop_duplicates(keep='first')
#
# X = df.drop(columns=['Is_laundering'])
# y = df['Is_laundering'] #target
# X['Time'] = pd.to_datetime(X['Time'], format='%H:%M:%S')
# X['Date'] = pd.to_datetime(X['Date'], format='%Y-%m-%d')
# X['Year'] = X['Date'].dt.year
# X['Month'] = X['Date'].dt.month
# X['Day'] = X['Date'].dt.day
# X['Hour'] = X['Time'].dt.hour
# X['Minute'] = X['Time'].dt.minute
# X['Second'] = X['Time'].dt.second
# X = X.drop(columns=['Date', 'Time'])
#
# categorical_columns = [
#     'Payment_currency',
#     'Received_currency',
#     'Sender_bank_location',
#     'Receiver_bank_location',
#     'Payment_type',
#     'Laundering_type'
# ]
# X = pd.get_dummies(X, columns=categorical_columns)
# smote = SMOTE(sampling_strategy=0.3, random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)
# data = pd.DataFrame(X_resampled, columns=X.columns)
# data['Is_laundering'] = y_resampled
#
# high_risk_countries = {"Nigeria", "Morocco", "Turkey", "Pakistan", "Mexico", "Spain"}
# # Identify one-hot encoded columns for high-risk countries
# sender_columns = [f"Sender_bank_location_{country}" for country in high_risk_countries]
# receiver_columns = [f"Receiver_bank_location_{country}" for country in high_risk_countries]
#
# # Check if any of the sender or receiver columns indicate a high-risk country
# data['high_risk_countries'] = data[sender_columns + receiver_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
#
# # Identify boolean columns and convert them to integers
# bool_columns = data.select_dtypes(include=['bool', 'object']).columns
#
# for col in bool_columns:
#     # Convert boolean-like object columns to actual boolean type first
#     if data[col].dtype == 'object':
#         data[col] = data[col].map({'True': True, 'False': False, 'true': True, 'false': False})
#
#     # Convert boolean columns to integers
#     data[col] = data[col].astype(int)
#
# if data['Is_laundering'].isna().sum() > 0:
#     data = data.dropna(subset=['Is_laundering'])