import json
import zipfile
import os
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG19
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras import Model, Sequential, metrics, optimizers
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D,\
                                    MaxPooling2D, UpSampling2D, GlobalAveragePooling2D,\
                                    Layer, Lambda,Flatten, Reshape, Conv2DTranspose,\
                                    Activation, LeakyReLU, Dropout, InputLayer

def encoder_network(input_shape, latent_dim=100):
    def sampling(args):
        z_mean, z_log_var = args
        epsilon_mean = 0
        epsilon_std = 1.0
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=epsilon_mean, stddev=epsilon_std)
        sampled_z = z_mean + K.exp(z_log_var / 2) * epsilon
        return sampled_z
    
    input_img = Input(shape=input_shape)
    x = Conv2D(32, 4, strides=(2, 2))(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(64, 4, strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, 4, strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 4, strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(512, 4, strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = GlobalAveragePooling2D()(x)

    z_mean = Dense(latent_dim, name='mean')(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling)([z_mean, z_log_var])

    encoder = Model(input_img, z)
    return encoder, z_mean, z_log_var

def decoder_network(latent_dim=100):
    decoder_input = Input(shape=(latent_dim,))
    x = Dense(4096)(decoder_input)
    x = Reshape((4, 4, 256))(x)

    x = UpSampling2D((2, 2), interpolation='nearest')(x)
    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = UpSampling2D((2, 2), interpolation='nearest')(x)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = UpSampling2D((2, 2), interpolation='nearest')(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = UpSampling2D((2, 2), interpolation='nearest')(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = UpSampling2D((2, 2), interpolation='nearest')(x)
    x = Conv2D(3, 3, strides=1, padding='same', activation='sigmoid')(x)

    decoder = Model(decoder_input, x)
    return decoder

def add_noise(im_as_array):
    img = Image.fromarray((im_as_array*255).astype(np.uint8))
    gaussian = np.random.normal(0, 50, (img.size[0],img.size[1], 3))
    noisy_img = img + gaussian
    return np.clip(np.array(noisy_img), 0, 255) / 255

def remove_border(im_as_array):
    img = Image.fromarray((im_as_array*255).astype(np.uint8))
    im_crop = ImageOps.crop(img, border=10)
    new_im = ImageOps.expand(im_crop,border=10,fill='black')
    return np.array(new_im) / 255

def rotate_im(im_as_array):
    img = Image.fromarray((im_as_array*255).astype(np.uint8))
    return np.array(img.rotate(10)) / 255

def display_recalled(x_test_new, decoded_imgs, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test_new[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
def crop_resize_normalize(item):
    img = item['image']
    # Normalise and resize image
    resized = tf.image.resize(tf.cast(img, tf.float32) / 255.0, [128, 128])
    item['image'] = resized
    return item

def preprocess(image):
    # Returns image, image because when training, vae expects same input image as output
    image = image['image']
    return image, image

def preprocess_test(image):
    return image['image']
