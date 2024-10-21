import pandas as pd
from imblearn.over_sampling import SMOTE


def preprocess(file_path, apply_smote=False):
    # Load the data
    df = pd.read_csv(file_path)
    df = df.sample(n=200000, random_state=1)  # Sample 'n' rows from the data
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

    # Apply SMOTE only if training data
    if apply_smote:
        smote = SMOTE(sampling_strategy=0.3, random_state=42)
        X, y = smote.fit_resample(X, y)

    # Create a new DataFrame from the data
    data = pd.DataFrame(X, columns=X.columns)
    data['Is_laundering'] = y

    # After applying get_dummies, check the columns
    # print(X.columns)  # This will help you see what columns were generated
    high_risk_countries = {"Nigeria", "Morocco", "Turkey", "Pakistan", "Mexico", "Spain"}
    # Identify one-hot encoded columns for high-risk countries
    sender_columns = [f"Sender_bank_location_{country}" for country in high_risk_countries]
    receiver_columns = [f"Receiver_bank_location_{country}" for country in high_risk_countries]
    data['high_risk_countries'] = data[sender_columns + receiver_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    # Convert boolean columns to integers
    bool_columns = data.select_dtypes(include=['bool', 'object']).columns
    for col in bool_columns:
        if data[col].dtype == 'object':
            data[col] = data[col].map({'True': True, 'False': False, 'true': True, 'false': False})
        data[col] = data[col].astype(int)

    # Handle any missing values in the target column
    if data['Is_laundering'].isna().sum() > 0:
        data = data.dropna(subset=['Is_laundering'])

    return data
