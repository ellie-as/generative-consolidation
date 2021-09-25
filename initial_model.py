import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def create_autoencoder():
    input = layers.Input(shape=(28, 28, 1))

    # Encoder
    filter_num = 8
    x = layers.Conv2D(filter_num, (3, 3), activation="relu", padding="same")(input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(500)(x)
    x = layers.Dropout(0.75)(x)
    x = layers.Dense(3136)(x)

    # Decoder
    x = layers.Reshape(target_shape=(14,14,16))(x)
    x = layers.Conv2DTranspose(filter_num, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer='adam', loss="binary_crossentropy")
    
    return autoencoder