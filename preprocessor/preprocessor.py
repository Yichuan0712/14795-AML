import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler


def split_dataset(file_path, train_size=0.9, test_size=0.1, val_size=0):
    """
    Splits the dataset into train, validation, and test sets while keeping the order unchanged.

    Parameters:
        file_path (str): The path to the dataset file.
        train_size (float): Proportion of the dataset to include in the training set.
        test_size (float): Proportion of the dataset to include in the test set.
        val_size (float, optional): Proportion of the dataset to include in the validation set. Default is 0.

    Returns:
        train_df (pd.DataFrame): Training dataset.
        val_df (pd.DataFrame or None): Validation dataset (if `val_size` > 0).
        test_df (pd.DataFrame): Test dataset.
    """
    # Load the full dataset
    df = pd.read_csv(file_path)

    # Ensure that the sum of train, val, and test sizes is equal to 1.0
    assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1.0"

    # Calculate the number of rows for each split
    train_end = int(train_size * len(df))
    val_end = int((train_size + val_size) * len(df))

    # Split the data without shuffling
    train_df = df[:train_end]

    # If validation size is 0, we skip the validation set
    if val_size > 0:
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
    else:
        val_df = None
        test_df = df[train_end:]

    return train_df, val_df, test_df


def preprocess(file_path, apply_smote=False):
    # Load the data
    df = pd.read_csv(file_path)

    # if n is not None and random_state is not None:
    #     df = df.sample(n=n, random_state=random_state)
    # elif n is not None and random_state is None:
    #     df = df.head(n)
    # elif n is None and random_state is not None:
    #     df = df.sample(frac=1, random_state=random_state)

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

    if apply_smote:
        smote = SMOTE(sampling_strategy=0.3, random_state=42)
        X, y = smote.fit_resample(X, y)

    data = pd.DataFrame(X, columns=X.columns)
    data['Is_laundering'] = y

    high_risk_countries = {"Nigeria", "Morocco", "Turkey", "Pakistan", "Mexico", "Spain"}
    sender_columns = [f"Sender_bank_location_{country}" for country in high_risk_countries]
    receiver_columns = [f"Receiver_bank_location_{country}" for country in high_risk_countries]
    data['high_risk_countries'] = data[sender_columns + receiver_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    bool_columns = data.select_dtypes(include=['bool', 'object']).columns
    for col in bool_columns:
        if data[col].dtype == 'object':
            data[col] = data[col].map({'True': True, 'False': False, 'true': True, 'false': False})
        data[col] = data[col].astype(int)

    if data['Is_laundering'].isna().sum() > 0:
        data = data.dropna(subset=['Is_laundering'])

    return data


def get_X_y_scaler(data, scaler=None):
    X = data.drop(columns=['Is_laundering'])
    y = data['Is_laundering']  # target
    if scaler is None:
        scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def get_X_y(data):
    X = data.drop(columns=['Is_laundering'])
    y = data['Is_laundering']
    return X, y
