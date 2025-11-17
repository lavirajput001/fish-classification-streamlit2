import tensorflow as tf
import pickle
import os

def load_model_and_encoder():
    model_path = "patternnet_model.h5"
    encoder_path = "label_encoder.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label Encoder not found: {encoder_path}")

    model = tf.keras.models.load_model(model_path)

    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder
