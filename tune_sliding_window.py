"""
Script for tuning the sliding window algorithm, finishing the model which was
trained earlier using the script train_model.py

INPUT: The path to the logdir of the corresponding model/experiment.
"""

import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import glob, os
import datetime
from shutil import copyfile
import argparse

import data.data_helpers as data_helpers
import models.model_helpers as model_helpers
import models.tuning_helpers as tuning_helpers
import models.training_vis as training_vis
import json

from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet_v2
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50



def main(logdir, colab):

    print(f"Continue tuning the window algorithm in {logdir}...")

    # Get summary from pre-training
    with open(os.path.join(logdir, 'results_classifier.json'), 'r') as f:
        results_classifier = json.load(f)

    # get parameter dictionary
    PARAMS = results_classifier['params']

    # update parameter dictionary (only necessary if trained on colab before!)
    #colab = False
    if colab:
        new_data_directories = ['data/FIDS30',
                                'data/original_clean',
                                'data/update_9sep',
                                'data/marianne_update',
                                'data/data_gleb_upsample']

        new_real_validation_path = 'data/validation_real'
        new_artificial_validation_path = 'data/validation_artificial'

        PARAMS.update({'data_directories': new_data_directories,
                       'real_validation_path': new_real_validation_path,
                       'artificial_validation_path': new_artificial_validation_path})


    classes = results_classifier['classes']
    class2ind = results_classifier['class2ind']
    ind2class = results_classifier['ind2class']
    ind2class = {int(k): v for (k,v) in ind2class.items()}

    logging = True
    saving = True
    log_image_stats = False

    #### Tune the sliding window algorithm

    # get validation set with "real" fridge scenes
    val_real_data = data_helpers.get_validation_dict(PARAMS['real_validation_path'], classes, verbose=0)

    # get validation set with "artificial" fridge scenes
    val_artificial_data = data_helpers.get_validation_dict(PARAMS['artificial_validation_path'], classes, verbose=0)

    # show example pictures
    fig = training_vis.show_example_picures_validation(val_real_data, 15,
                                                       logging=logging,
                                                       logpath=os.path.join(logdir, 'figures/example_validation_real.png'))

    fig = training_vis.show_example_picures_validation(val_artificial_data, 15,
                                                       logging=logging,
                                                       logpath=os.path.join(logdir, 'figures/example_validation_artificial.png'))


    # load classifier models
    model = load_model(
        logdir,
        custom_objects=None,
        compile=True
    )

    # define preprocessing function
    if PARAMS['base_net'] == 'resnet50':
        preprocess_function = preprocess_resnet50
    elif PARAMS['base_net'] == 'mobilenet_v2':
        preprocess_function = preprocess_mobilenet_v2

    results = tuning_helpers.tuning_loop_sliding_window_tight(PARAMS['scaling_factors'], PARAMS['sliding_strides'], PARAMS['thr_list'], PARAMS['overlap_thr_list'],
                                                      val_real_data, val_artificial_data, ind2class, classes,
                                                      model, preprocess_function, PARAMS['target_size'][0],
                                                      log_image_stats=log_image_stats)


    # save tuning results to json-file
    if logging:
        with open(os.path.join(logdir, 'results_tuning.json'), 'w+') as f:
            json.dump(results, f)


    #### Select combination of hyperparameters with highest F1-score

    # get summary metrics for each set of sliding window parameters
    metrics_df = pd.DataFrame(results)
    metrics_df.scaling_factors = metrics_df.scaling_factors.astype(str)

    metrics_df = metrics_df.groupby(['data_type', 'thr', 'overlap_thr', 'scaling_factors'])['precision', 'recall'].mean()
    metrics_df['f1'] = 2*metrics_df.precision*metrics_df.recall/(metrics_df.precision + metrics_df.recall)
    metrics_df['f1'] = metrics_df['f1'].fillna(0)
    metrics_df = metrics_df.reset_index()

    # save summary metrics evaluation to json
    if logging:
        metrics_df.to_json(os.path.join(logdir, 'metrics_df.json'))

    # aggregate metrics per type of dataset
    metrics_per_dataset = metrics_df.pivot_table(index=['thr', 'overlap_thr', 'scaling_factors'],
                                           columns=['data_type'],
                                           values=['f1', 'precision', 'recall'])
    new_columns = [a + '_' + b for (a,b) in metrics_per_dataset.columns]
    metrics_per_dataset.columns = new_columns
    metrics_per_dataset['f1'] = (metrics_per_dataset['f1_artificial'] +  metrics_per_dataset['f1_real'])/2.
    metrics_per_dataset['precision'] = (metrics_per_dataset['precision_artificial'] +  metrics_per_dataset['precision_real'])/2.
    metrics_per_dataset['recall'] = (metrics_per_dataset['recall_artificial'] +  metrics_per_dataset['recall_real'])/2.


    # get optimal parameters (optimizing f1 score)
    opt_thr, opt_overlap_thr, opt_scaling_factors = metrics_per_dataset.loc[metrics_per_dataset.idxmax()['f1']].name

    # get optimal scaling factors as list and find corresponding sliding strides
    opt_scaling_factors = eval(opt_scaling_factors) # transform from str to list
    opt_sliding_strides = [PARAMS['sliding_strides'][PARAMS['scaling_factors'].index(f)] for f in opt_scaling_factors]


    tuning_final = {'opt_thr': opt_thr,
                    'opt_overlap_thr': opt_overlap_thr,
                    'opt_scaling_factors': opt_scaling_factors,
                    'opt_sliding_strides': opt_sliding_strides,
                    }
    tuning_final.update(metrics_per_dataset.loc[metrics_per_dataset.idxmax()['f1']].to_dict())

    # save final parameters to json file
    if logging:
        pd.Series(tuning_final).to_json(os.path.join(logdir, 'tuning_final.json'))

    # show final parameters
    print("Tuned parameters:")
    print(pd.Series(tuning_final))

    #### Generate summary figures

    fig = training_vis.scatter_precision_recall(metrics_df,
                               logging=logging,
                               logpath=os.path.join(logdir, 'figures/precision_recall.png'))

    fig = training_vis.lines_f1_overlap_thr(metrics_df,
                                   logging=logging,
                                   logpath=os.path.join(logdir, 'figures/f1_overlap_thr.png'))

    fig = training_vis.bars_f1_scaling_factors(metrics_df,
                                   logging=logging,
                                   logpath=os.path.join(logdir, 'figures/f1_scaling_factors.png'))


    #### Generate example output pictures
    def generate_output_pictures(dataset, savepath, keys):
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
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                fig.savefig(os.path.join(savepath, f'real_{k}.png'), bbox_inches='tight')


    # save some results for artificial dataset
    savepath = os.path.join(logdir, 'figures', 'results', 'artificial')
    generate_output_pictures(val_artificial_data, savepath, keys=np.arange(0, 30, dtype='int'))

    # save results for real dataset
    savepath = os.path.join(logdir, 'figures', 'results', 'real')
    generate_output_pictures(val_real_data, savepath, keys=list(val_real_data.keys()))



###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("logdir", type=str,
                    help="Path to logdir for a given experiment.")
parser.add_argument("--colab", type=int, default=0,
                    help="Re-define paths from Google Drive to local drive")
args = parser.parse_args()

if __name__ == '__main__':

    logdir = str(args.logdir)
    colab = bool(int(args.colab))

    # execute main function
    main(logdir, colab)
