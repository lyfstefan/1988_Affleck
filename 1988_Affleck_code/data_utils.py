import os
import numpy as np
import time


def save_data(data, timestamp, folder="results/data", prefix="data"):
    os.makedirs(folder, exist_ok=True)
    filename = f"{prefix}_{timestamp}.npy"
    path = os.path.join(folder, filename)
    np.save(path, data)
    print(f"Data saved to {path}")
    return path

def load_data(path):
    data = np.load(path, allow_pickle=True)
    print(f"Data loaded from {path}")
    return data