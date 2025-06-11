import math
import numpy as np
import os
import json

_PI_180 = math.pi / 180.0
_180_PI = 180.0 / math.pi
R = 6371009


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth specified in decimal degrees.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def distance_lat_lon(lat1, lon1, lat2, lon2):
    """
    Calculate the distance in meters between two points specified by latitude and longitude.
    """
    C = haversine_distance(lat1, lon1, lat2, lon2)
    u = C * math.cos(lat1 * _PI_180)
    v = C * math.sin(lat1 * _PI_180)
    return u, v


def load_data(data_dir="Data/Simplified_Data"):
    files = os.listdir(data_dir)
    files = [
        file for file in files if file.endswith(".json") and not file.startswith("._")
    ]
    datasets = []
    for file in files[::]:
        data = []
        print(f"Loading {file}...", end="\r")
        with open(os.path.join(data_dir, file), "r") as f:
            raw_data = json.load(f)
            for entry in raw_data:
                data.append(entry)
        datasets.append(data)
    return datasets


def entry_to_array(entry):
    gh = np.array(entry["gh_interp"])
    u = np.array(entry["u_interp"])
    v = np.array(entry["v_interp"])
    w = np.array(entry["w_interp"])
    point_1 = np.array(entry["point_1"])
    point_2 = np.array(entry["point_2"])

    lat = point_1[0]
    lon = point_1[1]
    alt = point_1[2]

    r = R + alt

    relative_point = np.array(point_2, dtype=np.float32) - np.array(
        point_1, dtype=np.float32
    )

    dt = 60
    dx = u * dt  # vers l'est
    dy = v * dt  # vers le nord

    # Conversion mètres -> degrés
    delta_lat = (dy / r) * (180 / math.pi)
    delta_lon = (dx / (r * math.cos(math.radians(lat)))) * (180 / math.pi)

    sequence_features = np.array(
        [
            lat,
            lon,
            # alt / 30000,
            delta_lat/(120*dt),
            delta_lon/(120*dt),
        ],
        dtype=np.float32,
    )
    return sequence_features, relative_point[:2]


def create_sequences(data, window_size=30):
    y_all_datasets = []
    X_all_datasets = []

    for dataset_idx, dataset in enumerate(data):
        X_dataset_features = []
        y_dataset_targets_for_windowing = []

        if not dataset:
            print(f"Warning: Dataset {dataset_idx} is empty.")
            continue

        for entry in dataset:
            features, target = entry_to_array(entry)
            X_dataset_features.append(features)
            y_dataset_targets_for_windowing.append(target)

        if not X_dataset_features:
            print(
                f"Warning: Dataset {dataset_idx} resulted in no features after entry_to_array."
            )
            continue

        X_dataset_features_np = np.array(X_dataset_features)
        y_dataset_targets_np = np.array(y_dataset_targets_for_windowing)

        if len(X_dataset_features_np) <= window_size:
            print(
                f"Warning: Dataset {dataset_idx} has {len(X_dataset_features_np)} timesteps, not enough for window_size {window_size}. Skipping."
            )
            continue

        X_sequences_for_dataset = []
        y_targets_for_dataset = []
        for i in range(len(X_dataset_features_np) - window_size):
            sequence_input = np.array(X_dataset_features_np[i : i + window_size])
            target_output = np.array(y_dataset_targets_np[i + window_size - 1])

            sequence_input = sequence_input - np.array(
                [*sequence_input[-1, :2], 0, 0]
            )  # Normalize to the last point in the sequence

            # si y = [0, 0] alors on ne garde pas la sequence
            if np.all(target_output == 0):
                continue

            X_sequences_for_dataset.append(sequence_input)
            y_targets_for_dataset.append(target_output)

        if X_sequences_for_dataset:
            X_all_datasets.extend(X_sequences_for_dataset)
            y_all_datasets.extend(y_targets_for_dataset)

    if not X_all_datasets:
        raise ValueError("No sequences were created. Check data and window_size.")

    return np.array(X_all_datasets, dtype=np.float32), np.array(
        y_all_datasets, dtype=np.float32
    )


def make_data(data_dir="Data/Simplified_Data", window_size=30):
    print(f"Loading raw data from {data_dir}...")
    data = load_data(data_dir)
    print(f"Creating sequences with window_size={window_size}...")
    X, y = create_sequences(data, window_size=window_size)

    os.makedirs("Data/NPY", exist_ok=True)

    np.save("Data/NPY/X_unified.npy", X)
    np.save("Data/NPY/y.npy", y)
    print(f"Data saved: X_unified.npy {X.shape}, y.npy {y.shape}")
    return X, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X, y = make_data(data_dir="Data/Simplified_Minute", window_size=10)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(X[0], y[0])

    # affiche tout les points dans la premier valeur de X puis le point de y correpondant
    for i in range(X.shape[1])[:]:
        plt.scatter(X[0, i, 0], X[0, i, 1], color="blue", label="Input Points")

    plt.scatter(y[0, 0], y[0, 1], color="red", label="Target Points", marker="x")
    plt.xlabel("Latitude (degrees)")
    plt.ylabel("Longitude (degrees)")
    plt.legend()
    plt.show()
