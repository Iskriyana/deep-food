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


def predict_on_whole_dataset(model, data_iterator, ind2class):
    """
    Generates prediciton on a given keras DataIterator object.
    Iterates through the whole dataset and makes predictions for each item.

    Args:
        model: Trained model
        data_iterator: Input data
        ind2class: dict-mapping from indices to labels

    Returns:
        images: all images as a list
        labels: all one-hot labels as a list
        predict_i: integer-encoded predictions
        predict_proba: associated confidence levels
        predict_labels: predicted classes
        ind_misclassified: indices of misclassified items
    """

    # get all images and labels in the given dataset
    images = []
    labels = []
    for _ in range(len(data_iterator)):
        imgs_, lbls_ = next(data_iterator)
        images.append(imgs_)
        labels.append(lbls_)

    images = np.vstack(images)
    labels = np.vstack(labels)

    # get probabilites for each class
    predict_proba = model.predict(images)

    # get class predictions from maximum probabilites
    predict_i = predict_proba.argmax(axis=1)
    predict_labels = [ind2class[i] for i in predict_i]

    # reduce array of probability to the probability of the predicted class
    predict_proba = predict_proba.max(axis=1)

    # get indices of misclassified examples
    ind_misclassified = np.nonzero(predict_i != np.nonzero(labels)[1])[0]

    return images, labels, predict_i, predict_proba, predict_labels, ind_misclassified


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


def visualize_predictions(img, predictions, probabilities, x0, y0, windowsize):
    """
    Generate visualization of predicted labels for bounding boxes.

    Args:
        img: input image
        predictions: predicted labels
        probabilities: associated confidence levels
        x0: x-coordinate top left corner
        y0: y-coordinate top left corner
        windowsize: list of sizes for each window (square windows)

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
            rect = patches.Rectangle((x,y), windowsize[i], windowsize[i], linewidth=2, edgecolor='r', facecolor='none')
            plt.text(x+5, y+20, predictions[i] + f'/{probabilities[i]:.2f}', fontsize=10, bbox=dict(facecolor='red', alpha=0.5, edgecolor='r'))

            # Add the patch to the Axes
            ax.add_patch(rect)

    return fig


def gen_img_pyramid(img, fx=[1.0, 0.75,0.5], fy=[1.0, 0.75,0.5]):
    """
    Generate differently scaled versions of the same image:

    Args:
        img: Target image
        fx: list of re-scaling factors in x-direction
        fy: list of re-scaling factors in y-direction

    Returns:
        image_pyramid: list of re-scaled images
    """
    assert len(fx)==len(fy)

    image_pyramid = []
    for fx_, fy_ in zip(fx, fy):
        new_img = cv2.resize(img, (0,0), fx=fx_, fy=fy_)
        image_pyramid.append(new_img)

    return image_pyramid


def pyramid_prediction(model, preprocess_func, img, scaling_factors=[1.0, 0.75, 0.5], thr=0.9, kernel_size=224, strides=[64, 64, 32]):
    """
    Predict on image pyramid using the sliding window approach.

    Args:
        model: loaded model
        preprocess_func: function for image pre-processing
        img: given image
        scaling_factors: list of scaling factors for pyramid
        thr: threshold level for confidence required to give a prediction
        kernel_size: size of the window square kernel
        strides: list of strides for the sliding windows

    Returns:
        image_pyramid: list of rescaled images
        pyramid_predictions: list of prediction arrays for each scaling factor
        pyramid_probabilities: list of probability arrays for each scaling factor
        pyramid_x0: list of x-coordinates for each scaling factor
        pyramid_y0: list of y-coordinates for each scaling factor
        pyramid_windowsize: list of sizes for each window (square windows)
    """

    image_pyramid = gen_img_pyramid(img, fx=scaling_factors, fy=scaling_factors)


    pyramid_predictions = []
    pyramid_probabilities = []
    pyramid_x0 = []
    pyramid_y0 = []
    pyramid_windowsize = []
    for i, img_ in enumerate(image_pyramid):
        predictions, probabilities, x0, y0 = sliding_prediction(model,
                                                                preprocess_func,
                                                                img_,
                                                                thr=thr,
                                                                kernel_size=kernel_size,
                                                                stride=strides[i])

        # re-scale positions to fit to the original image
        f = scaling_factors[i]
        x0 = x0/f
        y0 = y0/f
        window_size = int(kernel_size/f)

        pyramid_predictions.append(predictions)
        pyramid_probabilities.append(probabilities)
        pyramid_x0.append(x0)
        pyramid_y0.append(y0)
        pyramid_windowsize.append([window_size]*len(predictions))

    return image_pyramid, pyramid_predictions, pyramid_probabilities, pyramid_x0, pyramid_y0, pyramid_windowsize


def intersection_over_union_from_boxes(boxA, boxB):
    """
    Calculates the IoU score given two boxes.
    (Source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

    Args:
        boxA: (x1,y1,x2,y2) of box A
        boxB: (x1,y1,x2,y2) of box B

    Returns:
        iou: IoU score
    """

	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 0) * max(0, yB - yA + 0)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def nonmax_suppression(pred_labels, probabilities, x0, y0, windowsize, overlap_thr=0.1):
    """
    Run nonmax suppression algorithm on bounding boxes.
    Delete "other" boxes from predictions.

    Args:
        pred_labels: List of labels
        probabilities: List of probabilities
        x0: list of x-coordinates
        y0: list of y-coordinates
        windowsize: list of size of boxes (square boxes)
        overlap_thr: Maximal overlap between boxes in IoU

    Returns:
        new_pred_labels: clean list of lables
        new_probabilities: associated probabilities
        new_x0: corresponding x-coordinates
        new_y0: corresponding y-coordinates
    """

    # define list of proposals as list of indices over all predictions
    proposals = np.arange(0, len(pred_labels), dtype='int')

    # intialize final list of boxes
    final = []

    # delete all boxes labeled as "other"
    mask_other = [pred!='other' for pred in pred_labels]
    proposals = list(proposals[mask_other])

    while len(proposals)>0:

        # add the box with the highest confidence to the final selection
        ind_max = probabilities[proposals].argmax()
        select = proposals.pop(ind_max)
        final.append(select)

        # delete all boxes which overlap substantially with this last selected box
        delete_i = []
        for i, p in enumerate(proposals):

            # compute IoU score
            boxA = (x0[select], y0[select], x0[select]+windowsize[select], y0[select]+windowsize[select])
            boxB = (x0[p], y0[p], x0[p]+windowsize[p], y0[p]+windowsize[p])
            iou = intersection_over_union_from_boxes(boxA, boxB)

            if iou >= overlap_thr:
                delete_i.append(i)

        # update proposal list
        proposals = [proposals[i] for i in range(len(proposals)) if i not in delete_i]


    new_pred_labels = np.array(pred_labels)[final]
    new_probabilities = np.array(probabilities)[final]
    new_x0 = np.array(x0)[final]
    new_y0 = np.array(y0)[final]
    new_windowsize = np.array(windowsize)[final]

    return new_pred_labels, new_probabilities, new_x0, new_y0, new_windowsize
