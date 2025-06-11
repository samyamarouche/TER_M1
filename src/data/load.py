import numpy as np


def load_data():
    """
    Charge les données X et y pré-traitées (fenêtrées) à partir des fichiers NPY.
    La window_size est implicite dans la structure de X.npy.
    """
    print("Loading pre-processed data from NPY files...")
    try:
        X = np.load("Data/NPY/X_unified.npy")
        y = np.load("Data/NPY/y.npy")
        print(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
    except FileNotFoundError:
        print("Error: NPY files not found. Please generate them first using src.data.generate.make_data().")
        raise
        
    return X, y
