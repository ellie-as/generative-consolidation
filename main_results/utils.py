import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow_datasets as tfds
from PIL import Image
from config import models_dict, dims_dict


def preprocess(array):
    #Normalizes the supplied array and reshapes it into the appropriate format.
    array = array.astype("float64") / 255.0
    array = np.expand_dims(array, axis=3)
    return array


def noise_gaussian(array, noise_factor=0.4, seed=None):
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


def noise(array, noise_factor=0.4, seed=None):
    #Replace a fraction noise_factor of pixels with 0
    if seed is not None:
        np.random.seed(seed)
    shape = array.shape
    array = array.flatten()
    indices = np.random.choice(np.arange(array.size), replace=False,
                           size=int(array.size * noise_factor))
    array[indices] = 0
    array = array.reshape(shape)
    return np.clip(array, 0.0, 1.0)


def display(array1, array2, seed=None, title='Inputs and outputs of the model', dataset='mnist'):
    hopfield=False
    
    dim = array1[0].shape[0]
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
            plt.imshow(image1.reshape(dim, dim), cmap='binary', vmin=-1, vmax=1)
        else:
            plt.imshow(image1.reshape(dim, dim))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        if hopfield==True:
            plt.imshow(image2.reshape(dim, dim), cmap='binary', vmin=-1, vmax=1)
        else:
            plt.imshow(image2.reshape(dim, dim))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    fig.suptitle(title)
    plt.show()
    return fig

def load_cifar():
    dim = dims_dict['cifar']
    (train_data, train_labels), (test_data, _) = cifar10.load_data()
    new_train_data = np.empty([train_data.shape[0], dim[0], dim[0], 1])
    for ind, t in enumerate(train_data):
        im = Image.fromarray(t).resize((dim[0],dim[0])).convert('L')
        im = np.expand_dims(im, axis=-1)
        new_train_data[ind] = np.asarray(im)
    new_test_data = np.empty([test_data.shape[0], dim[0], dim[0], 1])
    for ind, t in enumerate(test_data):
        im = Image.fromarray(t).resize((dim[0],dim[0])).convert('L')
        im = np.expand_dims(im, axis=-1)
        new_test_data[ind] = np.asarray(im)
    train_data = new_train_data
    test_data = new_test_data
    return train_data, test_data

def load_solids(dataset, num=5000):
    dim = dims_dict[dataset]
    if dataset == 'solids':
        dim = dims_dict['solids']
        ds = tfds.load('symmetric_solids', split='test', shuffle_files=True)
        ds_info = tfds.builder('symmetric_solids').info
        df = tfds.as_dataframe(ds.take(num), ds_info)
        df = df.sample(frac=1)
        new_train_data = np.empty([num, dim[0], dim[0]])
        test_data  = df['image']
        train_data = df['image']
        for ind, t in enumerate(train_data):
            im = Image.fromarray(t).resize((dim[0],dim[0])).convert('L')
            new_train_data[ind] = np.asarray(im)
        new_test_data = np.empty([test_data.shape[0], dim[0], dim[0]])
        for ind, t in enumerate(test_data):
            im = Image.fromarray(t).resize((dim[0],dim[0])).convert('L')
            new_test_data[ind] = np.asarray(im)
        train_data = new_train_data
        test_data = new_test_data
        return train_data, test_data
    if dataset == 'shapes3d':
        dim = dims_dict[dataset]
        ds = tfds.load(dataset, split='train[:5%]', shuffle_files=True)
        ds_info = tfds.builder(dataset).info
        df = tfds.as_dataframe(ds.take(num), ds_info)
        df = df.sample(frac=1)
        new_train_data = np.empty([num, dim[0], dim[0]])
        test_data  = df['image']
        train_data = df['image']
        for ind, t in enumerate(train_data):
            im = Image.fromarray(t).resize((dim[0],dim[0])).convert('L')
            new_train_data[ind] = np.asarray(im)
        new_test_data = np.empty([test_data.shape[0], dim[0], dim[0]])
        for ind, t in enumerate(test_data):
            im = Image.fromarray(t).resize((dim[0],dim[0])).convert('L')
            new_test_data[ind] = np.asarray(im)
        train_data = new_train_data
        test_data = new_test_data
        return train_data, test_data
    if dataset == 'plant_village':
        dim = dims_dict[dataset]
        ds = tfds.load(dataset, split='train', shuffle_files=True)
        ds_info = tfds.builder(dataset).info
        df = tfds.as_dataframe(ds.take(num), ds_info)
        df = df.sample(frac=1)
        new_train_data = np.empty([num, dim[0], dim[0]])
        test_data  = df['image']
        train_data = df['image']
        for ind, t in enumerate(train_data):
            im = Image.fromarray(t).resize((dim[0],dim[0])).convert('L')
            new_train_data[ind] = np.asarray(im)
        new_test_data = np.empty([test_data.shape[0], dim[0], dim[0]])
        for ind, t in enumerate(test_data):
            im = Image.fromarray(t).resize((dim[0],dim[0])).convert('L')
            new_test_data[ind] = np.asarray(im)
        train_data = new_train_data
        test_data = new_test_data
        return train_data, test_data
    if dataset == 'kmnist' or dataset=='fashion_mnist':
        dim = dims_dict[dataset]
        ds = tfds.load(dataset, split='test', shuffle_files=True)
        ds_info = tfds.builder(dataset).info
        df = tfds.as_dataframe(ds.take(num), ds_info)
        new_train_data = np.empty([num, dim[0], dim[0]])
        test_data  = df['image']
        train_data = df['image']
        for ind, t in enumerate(train_data):
            im = Image.fromarray(t.reshape((28,28))).resize((dim[0],dim[0]))
            new_train_data[ind] = np.asarray(im)
        new_test_data = np.empty([test_data.shape[0], dim[0], dim[0]])
        for ind, t in enumerate(test_data):
            im = Image.fromarray(t.reshape((28,28))).resize((dim[0],dim[0]))
            new_test_data[ind] = np.asarray(im)
        train_data = new_train_data
        test_data = new_test_data
        return train_data, test_data
    
    
def prepare_data(dataset, display=False, num=10000):
    if dataset == 'mnist':
        (train_data, _), (test_data, _) = mnist.load_data()
#     elif dataset == 'fashion_mnist':
#         (train_data, _), (test_data, _) = fashion_mnist.load_data()
    elif dataset == 'cifar':
        train_data, test_data = load_cifar()
    else:
        train_data, test_data = load_solids(dataset)

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