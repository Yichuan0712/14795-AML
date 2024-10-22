import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


def train_autoencoder(X, y, epochs, batch_size, encoder_decoder_path='demo/encoder_decoder_model.h5', encoder_path='demo/encoder_model.h5'):

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    input_dim = X_train.shape[1]
    encoding_dim = 8
    input_layer = Input(shape=(input_dim,))

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

    history = autoencoder.fit(X_train, X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_valid, X_valid))

    encoder_model = Model(inputs=input_layer, outputs=encoded)
    encoded_data = encoder_model.predict(X)

    encoded_features_df = pd.DataFrame(encoded_data, columns=[f'Encoded_Feature_{i+1}' for i in range(encoding_dim)])
    new_dataset = pd.concat([encoded_features_df, y.reset_index(drop=True)], axis=1)
    autoencoder.save(encoder_decoder_path)
    encoder_model.save(encoder_path)

    return new_dataset, autoencoder, encoder_model


def infer_autoencoder(X, encoder_path='demo/encoder_model.h5'):
    encoder_model = load_model(encoder_path)

    encoded_data = encoder_model.predict(X)

    encoding_dim = encoded_data.shape[1]
    encoded_features_df = pd.DataFrame(encoded_data, columns=[f'Encoded_Feature_{i + 1}' for i in range(encoding_dim)])

    return str(encoded_features_df)
