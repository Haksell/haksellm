import os
import pickle


def save_model(model, model_name):
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{model_name}.pkl")

    try:
        print(f"Saving model to {model_path}...")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print("Model saved successfully.")
        return model_path
    except IOError as e:
        print(f"Error saving model: {e}")
        raise


def load_model(model_name):
    model_path = os.path.join("models", f"{model_name}.pkl")

    try:
        print(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
    except IOError as e:
        print(f"Error loading model: {e}")
