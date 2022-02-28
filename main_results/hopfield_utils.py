import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from math import factorial
from utils import load_cifar, load_solids
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from hopfield_models import ContinuousHopfield, DenseHopfield, ClassicalHopfield
from config import models_dict, dims_dict

def create_hopfield(num, hopfield_type='continuous', dataset='mnist'):
    images = load_images_mnist(num, dataset=dataset)
    images_np = convert_images(images, 'gray-scale')
    images_masked_np = mask_image_random(images_np)
    images_np = [im_np.reshape(-1, 1) for im_np in images_np]
    images_masked_np = [im_np.reshape(-1, 1) for im_np in images_masked_np]

    n_pixel = dims_dict[dataset][0]
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

def create_hopfield_batch(num, labels, hopfield_type='continuous'):
    images = load_images_mnist_batch(num, labels)
    images_np = convert_images(images, 'gray-scale')
    images_masked_np = mask_image_random(images_np)
    images_np = [im_np.reshape(-1, 1) for im_np in images_np]
    images_masked_np = [im_np.reshape(-1, 1) for im_np in images_masked_np]

    n_pixel = dims_dict[dataset][0]
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
        

def load_images_mnist(num, dataset='mnist'):
    if dataset == 'mnist':
        (train_data, _), (test_data, _) = mnist.load_data()
#     elif dataset == 'fashion_mnist':
#         (train_data, _), (test_data, _) = fashion_mnist.load_data()
    elif dataset == 'cifar':
        train_data, test_data = load_cifar()
    else:
        train_data, test_data = load_solids(dataset)
    images = []
    for i in range(num):
        im_arr = train_data[i]
        im = Image.fromarray(im_arr)
        images.append(im)
    return images


def load_images_mnist_batch(num, labels, dataset='mnist'):
    if dataset == 'mnist':
        (train_data, train_labels), (test_data, _) = mnist.load_data()
#     elif dataset == 'fashion_mnist':
#         (train_data, train_labels), (test_data, _) = fashion_mnist.load_data()
    elif dataset == 'cifar':
        train_data, test_data = load_cifar()
    else:
        train_data, test_data = load_solids(dataset)
    images = []
    count = 0
    for i in range(train_data.shape[0]):
        if count < num:
            if train_labels[i] in labels:
                im_arr = train_data[i]
                im = Image.fromarray(im_arr)
                images.append(im)
                count += 1
        else:
            break
    return images

def convert_images(images, color_option, dataset='mnist'):
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
    """masks every pixel. Masking value is randomly +/-1 

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
                if np.random.rand() < 0.5:
                    im_masked[i][j] = -1
                else:
                    im_masked[i][j] = +1
        images_np_masked.append(im_masked)
    return images_np_masked

