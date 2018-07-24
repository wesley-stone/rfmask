import threading
import multiprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import numpy as np
from PIL import Image


# Processes Doom screen image to produce cropped and resized image.
def process_frame(frame):
    s = frame[10:-10,30:-30]
    s = scipy.misc.imresize(s,[84,84])

    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def combine_img_prediction(data, gt, pred, rgb=False):
    """
    Combines the data, grouth thruth and the prediction into one rgb image

    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor

    :returns img: the concatenated rgb image
    """
    ny = pred.shape[2]
    ch = data.shape[3]
    if rgb:
        img = np.concatenate((to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)),
                              to_rgb(crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1)),
                              to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    else:
        img = np.concatenate((crop_to_shape(data, pred.shape).reshape(-1, ny, ch),
                              crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1),
                              pred[..., 1].reshape(-1, ny, 1)), axis=1)

    return img

def combine_to_display(data, gt, pred):
    img = np.concatenate((data, gt, pred), axis=1)
    return img


def save_image(img, path):
    """
    Writes the image to disk
    """
    if np.amax(img) <= 1:
        img = img*255
    if len(img.shape) > 2:
        img = np.squeeze(img)
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300, 300], quality=90)


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny, channels]

    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].

    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1]) // 2
    offset1 = (data.shape[2] - shape[2]) // 2
    return data[:, offset0:(-offset0), offset1:(-offset1)]