"""Some helper functions for running the model."""

import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import cv2
import matplotlib.patches as patches


def image2d_as_strided(img, kernel_size=224, stride=32):
    """
    Transforms a 2D array into a stack of strided squares

    Args:
        img: 2D image array (no channels)
        kernel_size: size of square kernel
        stride: stride length

    Returns:
        new_img: array of shape (strided_height, strided_width, kernel_size, kernel_size)
    """

    shape = (img.shape[0] - kernel_size + 1,
             img.shape[1] - kernel_size + 1,
             kernel_size,
             kernel_size)

    strides = 2*img.strides
    new_img = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    new_img = new_img[::stride, ::stride]

    # get coordinates
    X,Y = np.meshgrid(np.arange(img.shape[1], dtype='int'),
                      np.arange(img.shape[0], dtype='int'))
    strides = 2*X.strides
    X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    strides = 2*Y.strides
    Y = np.lib.stride_tricks.as_strided(Y, shape=shape, strides=strides)
    X = X[::stride, ::stride, 0, 0]
    Y = Y[::stride, ::stride, 0, 0]
    #print(X.strides, img.strides)
    #print(X.shape, img.shape)

    return new_img, X, Y


def imageRGB_as_strided(img, kernel_size=224, stride=32):
    """
    Transforms a 3D image array into a stack of strided squares

    Args:
        img: 2D image array (no channels)
        kernel_size: size of square kernel
        stride: stride length

    """
    for ch in range(3):
        channel = img[:,:,ch]
        new_channel, x0, y0 = image2d_as_strided(channel, kernel_size=kernel_size, stride=stride)
        if ch == 0:
            new_img = np.zeros(new_channel.shape + (3,), dtype=np.uint8)
            new_img[:, :, :, :, 0] = new_channel
        else:
            new_img[:, :, :, :, ch] = new_channel

    return new_img, x0, y0


def predict_stack(model, preprocess_func, stack):
    """
    Predict labels on a given image stack

    Args:
        model: loaded model
        preprocess_func: function for image pre-processing
        stack: image stack

    Returns:
        predictions: integer-encoded class labels
        probabilites: associated probabilities
    """

    predict_proba = model.predict(preprocess_func(stack))
    predictions = predict_proba.argmax(axis=1)
    probabilities = predict_proba[np.arange(0, predict_proba.shape[0]), predictions]

    return predictions, probabilities


def sliding_prediction(model, preprocess_func, img, thr=0.9, kernel_size=224, stride=32):
    """
    Apply model on a sliding window over a given image.

    Args:
        model: loaded model
        preprocess_func: function for image pre-processing
        img: given image
        thr: threshold level for confidence required to give a prediction
        kernel_size: size of the window square kernel
        stride: stride of the sliding window

    Returns:
        predictions: integer-encoded class labels
        probabilites: associated probabilities
        x0: x-coordinates of the predictions
        y0: y-coordinates of the predictions
    """

    # re-format image as stack of strided patches
    new_img, x0, y0 = imageRGB_as_strided(img, kernel_size=kernel_size, stride=stride)

    stack = new_img.reshape((-1, kernel_size, kernel_size, 3))
    x0 = x0.flatten()
    y0 = y0.flatten()

    # predict on image stack
    predictions, probabilities = predict_stack(model, preprocess_func, stack)
    mask = probabilities > thr

    predictions = predictions[mask]
    probabilities = probabilities[mask]
    x0 = x0[mask]
    y0 = y0[mask]

    return predictions, probabilities, x0, y0


def visualize_predictions(img, predictions, probabilities, x0, y0, kernel_size):
    """
    Generate visualization of predicted labels for bounding boxes.

    Args:
        img: input image
        predictions: predicted labels
        probabilities: associated confidence levels
        x0: x-coordinate top left corner
        y0: y-coordinate top left corner
        kernel_size: size of the sliding window kernel

    Returns:
        fig: Figure containing the visualization
    """

    # show image
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

    # superimpose boxes
    for i, (x,y) in enumerate(zip(x0,y0)):
        if (predictions[i] != "other"):

            # Create a Rectangle patch
            rect = patches.Rectangle((x,y), kernel_size, kernel_size, linewidth=2, edgecolor='r', facecolor='none')
            plt.text(x+5, y+20, predictions[i] + f'/{probabilities[i]:.2f}', fontsize=10, bbox=dict(facecolor='red', alpha=0.5, edgecolor='r'))

            # Add the patch to the Axes
            ax.add_patch(rect)

    return fig
