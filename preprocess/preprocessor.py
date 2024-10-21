df = pd.read_csv("/content/SAML-D.csv")
df = df.sample(n=200000, random_state=1)
df = df.drop_duplicates(keep = 'first')
from imblearn.over_sampling import SMOTE
X = df.drop(columns=['Is_laundering'])
y = df['Is_laundering'] #target
X['Time'] = pd.to_datetime(X['Time'], format='%H:%M:%S')
X['Date'] = pd.to_datetime(X['Date'], format='%Y-%m-%d')
X['Year'] = X['Date'].dt.year
X['Month'] = X['Date'].dt.month
X['Day'] = X['Date'].dt.day
X['Hour'] = X['Time'].dt.hour
X['Minute'] = X['Time'].dt.minute
X['Second'] = X['Time'].dt.second
X = X.drop(columns=['Date', 'Time'])
from sklearn.preprocessing import OneHotEncoder
categorical_columns = [
    'Payment_currency',
    'Received_currency',
    'Sender_bank_location',
    'Receiver_bank_location',
    'Payment_type',
    'Laundering_type'
]
X = pd.get_dummies(X, columns=categorical_columns)
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
data = pd.DataFrame(X_resampled, columns=X.columns)
data['Is_laundering'] = y_resampled

high_risk_countries = {"Nigeria", "Morocco", "Turkey", "Pakistan", "Mexico", "Spain"}
# Identify one-hot encoded columns for high-risk countries
sender_columns = [f"Sender_bank_location_{country}" for country in high_risk_countries]
receiver_columns = [f"Receiver_bank_location_{country}" for country in high_risk_countries]

# Check if any of the sender or receiver columns indicate a high-risk country
data['high_risk_countries'] = data[sender_columns + receiver_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)