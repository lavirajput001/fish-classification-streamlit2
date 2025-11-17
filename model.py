import tensorflow as tf
import json

def load_model_and_encoder():
    model = tf.keras.models.load_model("patternnet_model.h5")

    with open("label_encoder.pkl", "r") as f:
        label_encoder = json.load(f)

    return model, label_encoder
