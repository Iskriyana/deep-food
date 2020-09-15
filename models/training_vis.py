"""
Visualization routines for training
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import json
import models.model_helpers as model_helpers

def show_example_picures_validation(dataset, ind, logging=False, logpath=''):
    """
    Shows an example from a given validation dataset
    """
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(121)

    plt.imshow(dataset[ind]['image'])
    plt.xticks([])
    plt.yticks([])
    plt.text(0.05, 0.98, '\n'.join(dataset[ind]['labels']),
             transform = ax.transAxes, verticalalignment='top',
             bbox=dict(facecolor='red', alpha=0.5))

    if logging:
        fig.savefig(logpath, bbox_inches='tight')

    return fig


def scatter_precision_recall(metrics_df, logging=False, logpath=''):

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111)
    sns.scatterplot(data=metrics_df, x='recall', y='precision', hue='scaling_factors')
    plt.xlim([0,1.05])
    plt.ylim([0,1.05])
    ax.legend(title='scaling_factors', bbox_to_anchor=(1.05, 1), loc='upper left')

    if logging:
        fig.savefig(logpath, bbox_inches='tight')

    return fig


def lines_f1_overlap_thr(metrics_df, logging=False, logpath=''):

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111)
    sns.lineplot(data=metrics_df, x='overlap_thr', y='f1', hue='scaling_factors')
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax.legend(bbox_to_anchor=(1.05,1.0), title='scaling_factors')

    if logging:
        fig.savefig(logpath, logging=False, logpath='')

    return fig


def bars_f1_scaling_factors(metrics_df, logging=False, logpath=''):

    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(111)
    sns.barplot(data=metrics_df, x='scaling_factors', y='f1')
    sns.despine()

    if logging:
        fig.savefig(logpath, logging=False, logpath='')

    return fig


def generate_output_pictures(dataset, keys, logging=False, logpath='', prefix='real'):
    """Small helper function to be applied to both real and artificial dataset
    """

    for k in keys:
        img = dataset[k]['image']
        # perform object detection with final model
        pred_labels, probabilities, x0, y0, windowsize = model_helpers.object_detection_sliding_window(model,
                                                                                         img,
                                                                                         preprocess_function,
                                                                                         PARAMS['target_size'][0],
                                                                                         ind2class,
                                                                                         opt_scaling_factors,
                                                                                         opt_sliding_strides,
                                                                                         opt_thr,
                                                                                         opt_overlap_thr)

        # visualize results
        fig = model_helpers.visualize_predictions(img,
                                                  pred_labels,
                                                  probabilities,
                                                  x0,
                                                  y0,
                                                  windowsize)


        if logging:
            savepath = os.path.join(logpath, 'figures', 'output', prefix)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            fig.savefig(os.path.join(savepath, f'real_{k}.png'), bbox_inches='tight')
