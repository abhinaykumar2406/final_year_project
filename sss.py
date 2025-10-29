import socket
import threading
import pickle
import struct
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
import tensorflow as tf

# ======================= SETTINGS =======================
HOST = '127.0.0.1'
PORT = 5000
no_of_clients = 5  # change as needed

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
    x = layers.Dense(5, activation='softmax')(x)
    return x

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

# ======================= FEDERATED AGGREGATION =======================
lock = threading.Lock()
client_auto = []
client_class = []
client_samples = []

count_auto = 0
count_class = 0

# ======================= CLIENT HANDLER =======================
def client_handler(conn, addr, auto_model, class_model):
    global client_auto, client_class, client_samples, count_auto, count_class
    print(f"[+] Connected: {addr}")

    try:
        # ================= Autoencoder Phase =================
        n_samples = recv_pickle(conn)  # client sends its local sample count
        send_pickle(conn, auto_model.get_weights())  # send current global autoencoder weights

        updated_auto = recv_pickle(conn)
        with lock:
            client_auto.append((updated_auto, n_samples))
            count_auto += 1

        # Wait until all clients finish autoencoder
        while count_auto < no_of_clients:
            pass

        # Aggregate weighted FedAvg autoencoder
        with lock:
            total_samples = sum(s for _, s in client_auto)
            avg_auto = []
            for layers_tuple in zip(*[w for w, _ in client_auto]):
                layer_sum = sum(layer * (s / total_samples) for layer, (_, s) in zip(layers_tuple, client_auto))
                avg_auto.append(layer_sum)
            auto_model.set_weights(avg_auto)

        send_pickle(conn, auto_model.get_weights())  # send aggregated autoencoder back

        # ================= Classifier Phase =================
        n_samples = recv_pickle(conn)  # client sends its local sample count again
        send_pickle(conn, class_model.get_weights())  # send current global classifier weights

        updated_class = recv_pickle(conn)
        with lock:
            client_class.append((updated_class, n_samples))
            count_class += 1

        # Wait until all clients finish classifier
        while count_class < no_of_clients:
            pass

        # Aggregate weighted FedAvg classifier
        with lock:
            total_samples = sum(s for _, s in client_class)
            avg_class = []
            for layers_tuple in zip(*[w for w, _ in client_class]):
                layer_sum = sum(layer * (s / total_samples) for layer, (_, s) in zip(layers_tuple, client_class))
                avg_class.append(layer_sum)
            class_model.set_weights(avg_class)

        send_pickle(conn, class_model.get_weights())  # send aggregated classifier back
        print(f"[+] Client {addr} finished training")

    finally:
        conn.close()


# ======================= MAIN SERVER =======================
if __name__ == "__main__":
    input_shape = (187, 1)
    inp = keras.Input(shape=input_shape)
    autoencoder = keras.Model(inp, decoder(encoder(inp)))
    classifier = Model(inp, fc(encoder(inp)))
    autoencoder.compile(optimizer='RMSprop', loss='mse')
    classifier.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(no_of_clients)
    print(f"[📡] Server listening on {HOST}:{PORT} for {no_of_clients} clients...")

    while True:
        conn, addr = server.accept()
        t = threading.Thread(target=client_handler, args=(conn, addr, autoencoder, classifier), daemon=True)
        t.start()
