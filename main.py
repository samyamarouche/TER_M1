from src.data import load_data, make_data


from src.model import SimpleLSTM
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt


def main():
    window_size = 15
    num_features = 4

    print("🌍 GPS Trajectory Prediction with LSTM")
    print("=" * 50)

    npy_X_path = "Data/NPY/X_unified.npy"
    npy_y_path = "Data/NPY/y.npy"

    if not (os.path.exists(npy_X_path) and os.path.exists(npy_y_path)):
        print(
            f"📁 NPY data files not found. Generating data with window_size={window_size}..."
        )

        make_data(data_dir="Data/Simplified_Minute", window_size=window_size)
        print(f"✅ Generated unified data: {npy_X_path}, {npy_y_path}")
    else:
        print(f"Found existing NPY data files. Attempting to load them.")

    try:
        X, y = load_data()

        if X.shape[1] != window_size:
            print(
                f"⚠️ Warning: Loaded data has window_size {X.shape[1]}, but current setting is {window_size}."
            )
            print(f"Re-generating data with window_size={window_size}...")
            make_data(data_dir="Data/Simplified_Data", window_size=window_size)
            X, y = load_data()

        if X.shape[2] != num_features:
            raise ValueError(
                f"Loaded data has {X.shape[2]} features, but model expects {num_features}."
            )

        print(f"✅ Loaded data: X shape {X.shape}, y shape {y.shape}")
    except FileNotFoundError:
        print(
            "❌ Error: Data files could not be loaded or generated. Please check paths and source data."
        )
        return
    except ValueError as e:
        print(f"❌ Error during data loading/validation: {e}")
        return

    print("\n📊 Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"📚 Train set: {X_train.shape[0]:,} samples")
    print(f"🧪 Test set: {X_test.shape[0]:,} samples")

    print("\n🏗️  Initializing PyTorch SimpleLSTM model...")

    model_pytorch = SimpleLSTM(
        window_size=X.shape[1],
        input_features=X.shape[2],
        batch_size=2** 6,  # 64
        use_dropout=True,
    )

    model_pytorch.summary()

    print(f"🎯 Window size: {window_size}")
    print(f"🔧 Progressive learning features:")
    print(f"   • Adaptive learning rate with cyclical warm-up")
    print(f"   • Progressive batch size increase")
    print(f"   • Dynamic dropout adjustment")
    print(f"   • Precision-focused loss function")
    print(f"   • Adaptive early stopping")

    history = None
    try:
        print("Début de l'entraînement du modèle...")
        history = model_pytorch.fit(
            X_train, y_train, epochs=100, validation_split=0.2,
        )
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur.")
    finally:
        print("Fin de la phase d'entraînement (ou de sa tentative).")

    loss = model_pytorch.evaluate(X_test, y_test)
    print(f"\n🎯 Final Test Results: MSE={loss[0]:.6f}, Distance={loss[1]:.2f}m")

    if history is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(history["loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()
