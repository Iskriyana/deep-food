"""Some helper functions for tuning the sliding window algorithm"""
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import models.model_helpers as model_helpers


def tuning_loop_sliding_window(scaling_factors, sliding_strides,
                               val_real_data, val_artificial_data, ind2class,
                               model, preprocess_func, kernel_size):
    """
    Runs the sliding window on all images in both datasets with a threshold of 0.0
    and saves the results in a dict.
    """

    # initialize results logging
    results = []
    thr = 0.0 # take all predictions which are made (threshold will be applied later)
    for data_type in ['real', 'artificial']:
        print(f'Run on {data_type} data...')

        if data_type == 'real':
            dataset = val_real_data
        elif data_type == 'artificial':
            dataset = val_artificial_data

        n = len(dataset)
        #n = 10
        for i_img in tqdm.tqdm(range(n)):

            # select image to perform predictions on
            input_sample = dataset[i_img]
            img = input_sample['image']
            labels = input_sample['labels']

            # get predictions using the "pyramid_prediction" framework, but define only 1 scaling factor
            #
            # define larger scaling factors to detect smaller objects!
            #
            image_pyramid, pyramid_predictions, pyramid_probabilities, \
            pyramid_x0, pyramid_y0, pyramid_windowsize = model_helpers.pyramid_prediction(model,
                                                                      preprocess_func,
                                                                      img,
                                                                      scaling_factors=scaling_factors,
                                                                      thr=0.0,
                                                                      kernel_size=kernel_size,
                                                                      strides=sliding_strides)

            # get label names
            pyramid_pred_labels = [[ind2class[i] for i in predictions]
                           for predictions in pyramid_predictions]


            # save results in dict
            results.append({'data_type': data_type,
                            'i_img': i_img,
                            'actual_labels': labels,
                            'pyramid_pred_labels': pyramid_pred_labels,
                            'pyramid_probabilities': pyramid_probabilities,
                            'pyramid_x0': pyramid_x0,
                            'pyramid_y0': pyramid_y0,
                            'pyramid_windowsize': pyramid_windowsize})

    return results


def get_evaluation_metrics(actual_labels, pred_labels, classes):
    """
    Get evaluation metrics for one image.

    Args:
        actual_labels: Actual labels on the image
        pred_labels: Predicted labels for that image
        classes: List of all available classes

    Returns:
        precision: Precision
        recall: Recall
        TP: true positives
        FP: false positives
        TN: true negatives
        FN: false negatives
    """

    # true positives are intersection between predicted and actual labels
    TP = set(actual_labels).intersection(set(pred_labels))

    # false positives are difference between predicted and actual labels
    FP = set(pred_labels) - set(actual_labels)

    # false negatives are difference between actual and predicted labels
    FN = set(actual_labels) - set(pred_labels)

    # true negatives are intersection between the differences between all classes
    # and the actual and predicted classes respectively
    # (usually not so important for object detection)
    TN = (set(classes) - set(actual_labels)).intersection((set(classes) - set(pred_labels)))

    try:
        precision = len(TP) / (len(TP)+len(FP))
    except ZeroDivisionError:
        precision = 0.0
    recall  = len(TP) / (len(TP)+len(FN))
    accuracy = (len(TP) + len(TN))/(len(TP) + len(TN) + len(FP) + len(FN))

    return accuracy, precision, recall, TP, FP, TN, FN
