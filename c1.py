import socket
import pickle
import struct
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.models import Model
from sklearn.metrics import classification_report

# ======================= CONFIG =======================
HOST = '127.0.0.1'
PORT = 5000

# Client ID from command line
if len(sys.argv) > 1:
    client_id = int(sys.argv[1])
else:
    client_id = 1

# ======================= SOCKET HELPERS =======================
def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def recv_pickle(sock):
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    length = struct.unpack('>I', raw_len)[0]
    data = recvall(sock, length)
    return pickle.loads(data)

def send_pickle(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(struct.pack('>I', len(data)) + data)

# ======================= LOAD CLIENT DATA =======================
train_df = pd.read_csv(f'federated_clients/iid/client_{client_id}_train.csv', header=0)
test_df = pd.read_csv(f'federated_clients/iid/client_{client_id}_test.csv', header=0)

# Convert labels to integer
train_df.iloc[:, -1] = train_df.iloc[:, -1].astype(int)
test_df.iloc[:, -1] = test_df.iloc[:, -1].astype(int)

X_train = train_df.iloc[:, :-1].values
y_train = to_categorical(train_df.iloc[:, -1])

X_test = test_df.iloc[:, :-1].values
y_test = to_categorical(test_df.iloc[:, -1])

# ======================= ADD GAUSSIAN NOISE =======================
def add_gaussian_noise(signal):
    noise = np.random.normal(0, 0.01, signal.shape[0])
    return signal + noise

X_train_noise = np.array([add_gaussian_noise(x) for x in X_train])
X_test_noise = np.array([add_gaussian_noise(x) for x in X_test])

# Reshape for Conv1D
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
X_train_noise = X_train_noise.reshape(len(X_train_noise), X_train_noise.shape[1], 1)
X_test_noise = X_test_noise.reshape(len(X_test_noise), X_test_noise.shape[1], 1)

# Normalize
X_train_noise = (X_train_noise - X_train_noise.mean()) / X_train_noise.std()
X_test_noise = (X_test_noise - X_test_noise.mean()) / X_test_noise.std()

# ======================= MODEL ARCHITECTURE =======================
def encoder(input_data):
    x = layers.Conv1D(64, kernel_size=3, activation='relu')(input_data)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(4, kernel_size=3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(2, padding='same')(x)
    return encoded

def decoder(encoded):
    x = layers.Conv1D(4, kernel_size=3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(1, 3, activation='relu', padding='same')(x)
    decoded = layers.Cropping1D(cropping=(0, 1))(x)
    return decoded

def fc(enco):
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(enco)
    x = layers.Conv1D(64, kernel_size=1, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(118, activation='relu')(x)
    x = layers.Dense(y_train.shape[1], activation='softmax')(x)
    return x

# ======================= TRAIN CLIENT FUNCTION =======================
def train_client():
    input_shape = (X_train.shape[1], 1)
    inp = keras.Input(shape=input_shape)

    # Models
    autoencoder = keras.Model(inp, decoder(encoder(inp)))
    classifier = Model(inp, fc(encoder(inp)))

    autoencoder.compile(optimizer='RMSprop', loss='mse')
    classifier.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))

    # ---------------- Autoencoder Phase ----------------
    send_pickle(sock, len(X_train))  # send number of local samples
    auto_weights = recv_pickle(sock)  # receive global weights
    autoencoder.set_weights(auto_weights)

    autoencoder.fit(X_train_noise, X_train, epochs=10, batch_size=100, verbose=1)
    send_pickle(sock, autoencoder.get_weights())  # send updated weights

    aggregated_auto = recv_pickle(sock)  # receive aggregated global autoencoder
    autoencoder.set_weights(aggregated_auto)

    # ---------------- Classifier Phase ----------------
    send_pickle(sock, len(X_train))
    class_weights = recv_pickle(sock)
    classifier.set_weights(class_weights)

    classifier.fit(X_train_noise, y_train, epochs=10, batch_size=100, verbose=1)
    send_pickle(sock, classifier.get_weights())

    aggregated_class = recv_pickle(sock)
    classifier.set_weights(aggregated_class)

    # ---------------- EVALUATE ----------------
    y_pred = np.argmax(classifier.predict(X_test_noise), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(f"[Client {client_id}] Classification Report:")
    print(classification_report(y_true, y_pred))

    sock.close()

if __name__ == "__main__":
    train_client()
