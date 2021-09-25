import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def preprocess(array):
    #Normalizes the supplied array and reshapes it into the appropriate format.
    array = array.astype("float32") / 255.0
    array = np.expand_dims(array, axis=3)
    return array


def noise(array, noise_factor=0.4, seed=None):
    #Adds random noise to each image in the supplied array.
    if seed is not None:
        np.random.seed(seed)
    if array is not None:
        noisy_array = array + noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=array.shape
        )
    else:
        noisy_array = noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=(1, 28, 28)
        )
    return np.clip(noisy_array, 0.0, 1.0)


def display(array1, array2, seed=None, title='Inputs and outputs of the model'):
    hopfield=False
    #Displays ten random images from each one of the supplied arrays.
    if seed is not None:
        np.random.seed(seed)
        
    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    fig = plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        if hopfield==True:
            plt.imshow(image1.reshape(28, 28), cmap='binary', vmin=-1, vmax=1)
        else:
            plt.imshow(image1.reshape(28, 28))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        if hopfield==True:
            plt.imshow(image2.reshape(28, 28), cmap='binary', vmin=-1, vmax=1)
        else:
            plt.imshow(image2.reshape(28, 28))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    fig.suptitle(title)
    plt.show()
    return fig
    
    
def prepare_data(display=False):
    (train_data, _), (test_data, _) = mnist.load_data()

    # Normalize and reshape the data
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    # Create a copy of the data with added noise
    noisy_train_data = noise(train_data, noise_factor=0.6)
    noisy_test_data = noise(test_data, noise_factor=0.6)

    # Display the train data and a version of it with added noise
    if display == True:
        display(train_data, noisy_train_data)
    
    return train_data, test_data, noisy_train_data, noisy_test_data