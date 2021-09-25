import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from math import factorial
from tensorflow.keras.datasets import mnist
from hopfield_models import ContinuousHopfield, DenseHopfield, ClassicalHopfield

def create_hopfield(num, hopfield_type='continuous'):
    images = load_images_mnist(num)
    images_np = convert_images(images, 'black-white')
    images_masked_np = mask_image_random(images_np)
    images_np = [im_np.reshape(-1, 1) for im_np in images_np]
    images_masked_np = [im_np.reshape(-1, 1) for im_np in images_masked_np]

    n_pixel = 28
    orig_shape = n_pixel,n_pixel
    N = np.prod(orig_shape)
    train_patterns = images_np
    
    if hopfield_type == 'continuous':
        net = ContinuousHopfield(N, beta=10)
    if hopfield_type == 'dense':
        net = DenseHopfield(N, beta=10)
    if hopfield_type == 'classical':
        net = ClassicalHopfield(N)
        
    net.learn(train_patterns)
    return net 
        

def load_images_mnist(num):
    (train_data, _), (test_data, _) = mnist.load_data()
    images = []
    for i in range(num):
        im_arr = train_data[i]
        im = Image.fromarray(im_arr).convert('RGB')
        images.append(im)
    return images

def convert_images(images, color_option):
    """converts images to either binary black and white or gray-scale with pixel values
    in set {-1,+1} (for bw) or in intervall [-1,+1] (for gray-scale)

    Args:
        images ([PIL.Image]]): list of images loaded from PIL 
        color_option (str): either "black-white" or "gray-scale"

    Returns:
        [np.ndarray]: list of numpy arrays
    """
    valid_options = ['black-white', 'gray-scale']
    assert color_option in valid_options, 'unkown color option %s, Please choose from %s' % (color_option, valid_options)

    images_np = []
    for im in images:
        if color_option == 'black-white':
            im_grey = np.mean(im, axis=2)
            im_np = np.asarray(im_grey)
            im_np = np.where(im_np>128, 1, -1)
        elif color_option == 'gray-scale':
            im_grey = im.convert('L')
            im_np = np.asarray(im_grey)/255 *2 - 1
        images_np.append(im_np)
    return images_np

def mask_image_random(images_np):
    """masks every pixel with 50% chance. Masking value is randomly +/-1 

    Args:
        images_np ([np.ndarray]): list of images, each as numpy array

    Returns:
        [np.ndarray]: list of masked images as numpy arrays
    """
    images_np_masked = []
    n_pixel = images_np[0].shape[0]
    for im_np in images_np:
        im_masked = im_np.copy()
        for i in range(n_pixel):
            for j in range(n_pixel):
                if np.random.rand() < 1:
                    if np.random.rand() < 0.5:
                        im_masked[i][j] = -1
                    else:
                        im_masked[i][j] = +1
        images_np_masked.append(im_masked)
    return images_np_masked

