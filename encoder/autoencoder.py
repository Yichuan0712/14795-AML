import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
    print("Scaled data contains NaN or Infinite values")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
input_dim = X_train.shape[1]
encoding_dim = 8
input_layer = Input(shape=(input_dim,))
#we have created 3 layers to improve the encoding accuracy
encoder = Dense(64, activation='relu', kernel_initializer='he_normal')(input_layer)
encoder = Dense(32, activation='relu', kernel_initializer='he_normal')(encoder)
encoded = Dense(encoding_dim, activation='relu', kernel_initializer='he_normal')(encoder)
decoder = Dense(32, activation='relu', kernel_initializer='he_normal')(encoded)
decoder = Dense(64, activation='relu', kernel_initializer='he_normal')(decoder)
decoded = Dense(input_dim, activation='sigmoid')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.summary()
optimizer = Adam(learning_rate=0.0001)
autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
history=autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))
encoder_model = Model(inputs=input_layer, outputs=encoded)
encoded_data = encoder_model.predict(X_scaled)

if np.isnan(encoded_data).any() or np.isinf(encoded_data).any():
    print("Encoded data contains NaN or Infinite values")
encoded_features_df = pd.DataFrame(encoded_data, columns=[f'Encoded_Feature_{i+1}' for i in range(encoding_dim)])
new_dataset = pd.concat([encoded_features_df, y.reset_index(drop=True)], axis=1)
autoencoder.save('autoencoder_model.h5')
encoder_model.save('encoder_model.h5')