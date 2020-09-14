#!/usr/bin/env python
# coding: utf-8

"""
Script derived from notebook train_model.ipynb.
Specify hyperparameters in PARAMS dictionary at top-level.
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



def main(PARAMS):
    """
    Main training routine
    """

    # whether to log the results to json:
    logging = True

    # whether to save the model:
    saving  = True

    # define log directories
    timestamp =  datetime.datetime.now().strftime("%Y_%m_%d-%H:%M")
    logdir = f'logs/experiments/{PARAMS["experiment_name"]}_' + timestamp
    logdir_tb = f'logs/scalars/{PARAMS["experiment_name"]}_' + timestamp

    # create logging folder and tensorboard callback function
    if logging:
        print(f'Log results to {logdir}')
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        tensorboard_callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir_tb)]
    else:
        logdir = ''
        logdir_tb = ''
        tensorboard_callbacks = []

    # save a copy of the current script (for repeatability)
    if logging:
        src = os.path.realpath(__file__)
        suffix = f'{PARAMS["experiment_name"]}_' + timestamp
        dst = os.path.join(logdir, f'train_model_{suffix}.py')
        copyfile(src, dst)

    #### Load train and test data
    data_df_train, data_df_test, classes, class2ind, ind2class = \
            data_helpers.get_train_test_data_df(PARAMS['data_directories'], PARAMS['test_size'], PARAMS['seed'])

    #### Set up data pipeline
    #
    #  - Use keras-ImageDataGenerator for pre-processing and define data augmentations
    #  - Create keras-DataFrameIterators as an input to the model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet_v2
    from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # define preprocessing function
    if PARAMS['base_net'] == 'resnet50':
        preprocess_function = preprocess_resnet50
    elif PARAMS['base_net'] == 'mobilenet_v2':
        preprocess_function = preprocess_mobilenet_v2

    # crea ImageDataGenerator with data augmentations
    image_generator_train = ImageDataGenerator(horizontal_flip=True,
                                               vertical_flip=True,
                                               rotation_range=180,
                                               shear_range=30,
                                               zoom_range=[0.5,1],
                                               preprocessing_function=preprocess_function)

    image_generator_test = ImageDataGenerator(preprocessing_function=preprocess_function)

    # create data iterators
    train_iterator = data_helpers.get_data_frame_iterator(data_df_train, image_generator_train, PARAMS)
    test_iterator = data_helpers.get_data_frame_iterator(data_df_test, image_generator_test, PARAMS)


    #### Show some example pictures from the train set
    #
    #  - Use `deprocess_func=None` for MobileNet
    #  - Use `deprocess_func=data_helpers.deprocess_imagenet` for Resnet50

    # generate example batch
    image_batch, label_batch = next(train_iterator)

    # get correct image deprocessing function for proper visualization of images
    if PARAMS['base_net'] == 'mobilenet_v2':
        plot_deprocess_func = None
    else:
        plot_deprocess_func = data_helpers.deprocess_imagenet

    # generate figure
    fig = data_helpers.image_grid(image_batch, label_batch, ind2class, n_row=4, n_col=8, deprocess_func=plot_deprocess_func)

    if logging:
        if not os.path.exists(os.path.join(logdir, 'figures')):
            os.makedirs(os.path.join(logdir, 'figures'))
        fig.savefig(os.path.join(logdir, 'figures/training_examples.png'), bbox_inches='tight')

    with sns.axes_style('darkgrid'):
        fig = data_helpers.show_label_distribution(data_df_train, ind2class, 'Train')
        if logging:
            fig.savefig(os.path.join(logdir, 'figures/training_distribution.png'), bbox_inches='tight')

        fig = data_helpers.show_label_distribution(data_df_test, ind2class, 'Test')


    # ### Generate model
    #
    # Create the model using a transfer learning approach.
    #
    #  - Download pre-trained base model
    #  - Freeze all layers of the base model
    #  - Append a custom head using the Sequential API
    #  - Compile the model
    from tensorflow.keras.optimizers import Adam

    # download pre-trained model
    if PARAMS['base_net'] == 'resnet50':
        pretrained_model = tf.keras.applications.ResNet50(input_shape=PARAMS['target_size']+(3,),
                                                          include_top=False)
    elif PARAMS['base_net'] == 'mobilenet_v2':
        pretrained_model = tf.keras.applications.MobileNetV2(input_shape=PARAMS['target_size']+(3,),
                                                          include_top=False)

    # freeze pre-trained model
    pretrained_model.trainable = False

    model = tf.keras.Sequential([
        pretrained_model,
        *eval(PARAMS['head_net']),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])

    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=PARAMS['lr_cold']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    #### Train the head of the model (with the base frozen)

    steps_per_epoch = len(train_iterator.filenames)//PARAMS['batch_size']+1

    history_cold = model.fit_generator(
        train_iterator,
        validation_data=test_iterator,
    	epochs=PARAMS['epochs_cold'],
        steps_per_epoch=steps_per_epoch,
        callbacks=tensorboard_callbacks)


    #### Train the head of the model (with the base un-frozen)

    pretrained_model.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=PARAMS['lr_finetune']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_finetune = model.fit_generator(
        train_iterator,
        validation_data=test_iterator,
    	epochs=PARAMS['epochs_cold']+PARAMS['epochs_finetune'],
        steps_per_epoch=steps_per_epoch,
        callbacks=tensorboard_callbacks,
        initial_epoch=PARAMS['epochs_cold'])

    # append both histories
    history = {k: v + history_finetune.history[k] for (k,v) in history_cold.history.items()}

    # plot history
    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(121)
    plt.plot(history['loss'], 'b')
    plt.plot(history['val_loss'], 'r')
    plt.ylim([0, None])
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'])

    ax = plt.subplot(122)
    plt.plot(history['accuracy'], 'b')
    plt.plot(history['val_accuracy'], 'r')
    plt.ylim([0,1])
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'])

    if logging:
        fig.savefig(os.path.join(logdir, 'figures/loss_accuracy_training.png'), bbox_inches='tight')


    #### Look at results
    #
    #  - Generate predictions on the whole test set
    #  - Look at some predictions
    #  - Look at the most wrong predictions

    # predict on the whole set
    images, labels, predict_i, predict_proba, predict_labels, ind_misclassified = model_helpers.predict_on_whole_dataset(model, test_iterator, ind2class)

    # show first 32 images in test set
    fig = data_helpers.image_grid(images[:32], labels[:32], ind2class, n_row=4, n_col=8, deprocess_func=plot_deprocess_func,
                                  predict=predict_i[:32], predict_proba=predict_proba[:32], hspace=0.5)

    if logging:
        fig.savefig(os.path.join(logdir, 'figures/test_examples.png'), bbox_inches='tight')

    # show worst misclassifications
    ind_sorted = np.argsort(predict_proba[ind_misclassified])[::-1]
    ind_sorted = ind_misclassified[ind_sorted]

    fig = data_helpers.image_grid(images[ind_sorted], labels[ind_sorted], ind2class, n_row=4, n_col=8, deprocess_func=plot_deprocess_func,
                                  predict=predict_i[ind_sorted], predict_proba=predict_proba[ind_sorted], hspace=0.5)

    if logging:
        fig.savefig(os.path.join(logdir, 'figures/worst_test_examples.png'), bbox_inches='tight')


    #### Get confusion matrix and metrics
    #
    #  - Get model performance metrics
    #  - Get confusion matrix

    # generate evaluation dataframe
    eval_df = pd.DataFrame({'predicted': predict_i, 'actual': np.nonzero(labels)[1]})
    eval_df.predicted = eval_df.predicted.apply(lambda x: ind2class[x])
    eval_df.actual = eval_df.actual.apply(lambda x: ind2class[x])

    from sklearn.metrics import accuracy_score, precision_score, recall_score

    accuracy = accuracy_score(eval_df.predicted, eval_df.actual)
    micro_precision = precision_score(eval_df.predicted, eval_df.actual, average='micro')
    micro_recall = recall_score(eval_df.predicted, eval_df.actual, average='micro')
    macro_precision = precision_score(eval_df.predicted, eval_df.actual, average='macro')
    macro_recall = recall_score(eval_df.predicted, eval_df.actual, average='macro')

    print(f"Accuracy: {accuracy}\nMicro-precision: {micro_precision}\nMacro-precision: {macro_precision}\nMicro-recall: {micro_recall}\nMacro-recall: {macro_recall}\n")

    # confusion matrix
    from sklearn.metrics import confusion_matrix

    conf_matrix = pd.DataFrame(confusion_matrix(eval_df['actual'], eval_df['predicted'], labels=classes, normalize='true'),
                               index=classes, columns=classes)
    fig, ax = plt.subplots(figsize=(20,15))
    sns.heatmap((conf_matrix*100).astype(np.int), annot=True)

    if logging:
        fig.savefig(os.path.join(logdir, 'figures/confusion_matrix.png'), bbox_inches='tight')


    #### Log results and save the model
    from tensorflow.keras.models import save_model
    import json

    # define results dictionary
    results = {'params': PARAMS,
               'eval_df': eval_df.to_dict(),
               'history': history,
               'metrics': {'accuracy': accuracy,
                           'micro-precision': micro_precision,
                           'macro-precision': macro_precision,
                           'micro-recall': micro_recall,
                           'macro-recall': macro_recall,
               'model_path': logdir}}

    # save the model
    if saving:
        save_model(model, logdir)

    # save logging dict to json-file
    if logging:
        with open(os.path.join(logdir, 'results_classifier.json'), 'w+') as f:
            json.dump(results, f)


    #### Tune the sliding window algorithm

    ##### Get validation data

    # get validation set with "real" fridge scenes
    val_real_data = data_helpers.get_validation_dict(PARAMS['real_validation_path'], classes, verbose=0)

    # get validation set with "artificial" fridge scenes
    val_artificial_data = data_helpers.get_validation_dict(PARAMS['artificial_validation_path'], classes, verbose=0)

    # show examples pictures from validation set
    plt.figure(figsize=(12,12))

    ax = plt.subplot(121)
    plt.imshow(val_real_data[15]['image'])
    plt.xticks([])
    plt.yticks([])
    plt.text(0.05, 0.98, '\n'.join(val_real_data[15]['labels']),
             transform = ax.transAxes, verticalalignment='top',
             bbox=dict(facecolor='red', alpha=0.5))


    ax = plt.subplot(122)
    plt.imshow(val_artificial_data[15]['image'])
    plt.xticks([])
    plt.yticks([])
    plt.text(0.05, 0.98, '\n'.join(val_artificial_data[15]['labels']),
             transform = ax.transAxes, verticalalignment='top',
             bbox=dict(facecolor='red', alpha=0.5))


    #### Run sliding window algorithm on validation set
    #
    # The following cell runs the sliding window algorithm with the specified parameters over the whole validation set (both artificial and real) and saves intermediate results in a temporaray dataframe `sliding_df`.
    #
    # The algorithm does not yet apply thresholding or non-maximum suppression. This dataframe therefore contains the classification results for __ALL__ boxes (without thresholding).
    #
    #  - `scaling_factors` are the different scaling factors for the image pyramid
    #  - `sliding_strides` the are different strides for each level of the pyramid
    #
    # Using the results from this dataframe `sliding_df`, we can later perform thresholding and non-maximum suppression (which require a lot less computational power than the image classification itself) and find their optimal values.

    import models.tuning_helpers as tuning_helpers

    sliding_df = tuning_helpers.tuning_loop_sliding_window(scaling_factors=PARAMS['scaling_factors'],
                                            sliding_strides=PARAMS['sliding_strides'],
                                            val_real_data=val_real_data,
                                            val_artificial_data=val_artificial_data,
                                            ind2class=ind2class,
                                            model=model,
                                            preprocess_func=preprocess_function,
                                            kernel_size=PARAMS['target_size'][0])

    from itertools import combinations
    import tqdm

    # get all combinations of pyramid elements as list of index tuples
    N_pyramid = len(PARAMS['scaling_factors'])
    pyramid_combs = []
    for n in range(1, N_pyramid+1):
        pyramid_combs.extend(combinations(range(N_pyramid), n))


    # Iterate over all samples in both datasets, test all combinations of hyperparemeters
    # and evaluate the final performance of the object detection algorithm.
    tuning_df = []

    for sample in tqdm.tqdm(sliding_df):
        data_type = sample['data_type']
        # iterate over all values for the decision treshold
        for thr in PARAMS['thr_list']:
            # iterate over all values for the non-max suppresion threshold
            for overlap_thr in PARAMS['overlap_thr_list']:
                # iterate over all combination of pyramid levels / object sizes
                for comb_ind in pyramid_combs:


                    pred_labels, probabilities, x0, y0, windowsize = model_helpers.combine_pyramid_predictions(comb_ind,
                                                                  sample['pyramid_pred_labels'],
                                                                  sample['pyramid_probabilities'],
                                                                  sample['pyramid_x0'],
                                                                  sample['pyramid_y0'],
                                                                  sample['pyramid_windowsize'])

                    # apply decision threshold
                    mask = np.array(probabilities)>thr
                    pred_labels = pred_labels[mask]
                    probabilities = probabilities[mask]
                    x0 = x0[mask]
                    y0 = y0[mask]
                    windowsize = windowsize[mask]

                    # apply non-maximum suppression algorithm
                    pred_labels, probabilities, x0, y0, windowsize = model_helpers.nonmax_suppression(pred_labels,
                                                                                          probabilities,
                                                                                          x0,
                                                                                          y0,
                                                                                          windowsize,
                                                                                          overlap_thr=overlap_thr)

                    # get evaluation metrics
                    actual_labels = sample['actual_labels']
                    accuracy, precision, recall, TP, FP, TN, FN = tuning_helpers.get_evaluation_metrics(actual_labels, pred_labels, classes)

                    # log results
                    scaling_factors = np.array(PARAMS['scaling_factors'])[list(comb_ind)]
                    sliding_strides = np.array(PARAMS['sliding_strides'])[list(comb_ind)]

                    log = {'data_type': sample['data_type'],
                           'i_img': sample['i_img'],
                           'thr': thr,
                           'overlap_thr': overlap_thr,
                           'scaling_factors': scaling_factors.tolist(),
                           'sliding_strides': sliding_strides.tolist(),

                           'accuracy': accuracy,
                           'precision': precision,
                           'recall': recall,
                           'TP': list(TP),
                           'FP': list(FP),
                           'TN': list(TN),
                           'FN': list(FN),
                           #'actual_labels': list(actual_labels),
                           #'predicted_labels': pred_labels.tolist(),
                           #'probabilities': probabilities.tolist(),
                           #'x0': x0.tolist(),
                           #'y0': y0.tolist(),
                           #'windowsize': windowsize.tolist()
                           }

                    tuning_df.append(log)


    # save tuning results to json-file
    if logging:
        with open(os.path.join(logdir, 'results_tuning.json'), 'w+') as f:
            json.dump(tuning_df, f)


    #### Select combination of hyperparameters with highest F1-score

    # get summary metrics for each set of sliding window parameters
    metrics_df = pd.DataFrame(tuning_df)
    metrics_df.scaling_factors = metrics_df.scaling_factors.astype(str)

    metrics_df = metrics_df.groupby(['data_type', 'thr', 'overlap_thr', 'scaling_factors'])['precision', 'recall'].mean(0)
    metrics_df['f1'] = 2*metrics_df.precision*metrics_df.recall/(metrics_df.precision + metrics_df.recall)
    metrics_df = metrics_df.reset_index()

    # save summary metrics evaluation to json
    if logging:
        metrics_df.to_json(os.path.join(logdir, 'metrics_df.json'))

    # aggregate metrics per type of dataset
    metrics_per_dataset = metrics_df.pivot(index=['thr', 'overlap_thr', 'scaling_factors'],
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


    #### Generate some summary figures for the tuning process

    sns.set_context('talk')
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111)
    sns.scatterplot(data=metrics_df, x='recall', y='precision', hue='scaling_factors')
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax.legend(bbox_to_anchor=(1.7,1.0), title='scaling_factors')

    if logging:
        if not os.path.exists(os.path.join(logdir, 'figures')):
            os.makedirs(os.path.join(logdir, 'figures'))
        fig.savefig(os.path.join(logdir, 'figures/precision_recall.png'), bbox_inches='tight')


    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111)
    sns.lineplot(data=metrics_df, x='overlap_thr', y='f1', hue='scaling_factors')
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax.legend(bbox_to_anchor=(1.05,1.0), title='scaling_factors')

    if logging:
        fig.savefig(os.path.join(logdir, 'figures/f1_overlap_thr.png'), bbox_inches='tight')


    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(111)
    sns.barplot(data=metrics_df, x='scaling_factors', y='f1')
    sns.despine()

    if logging:
        fig.savefig(os.path.join(logdir, 'figures/f1_scaling_factors.png'), bbox_inches='tight')


    image_stats_df = pd.DataFrame(tuning_df)
    image_stats_df.scaling_factors = image_stats_df.scaling_factors.astype(str)

    image_stats_df = image_stats_df.set_index(['data_type', 'thr', 'overlap_thr', 'scaling_factors'])

    real_stats = {'TP': [], 'FP': [], 'TN': [], 'FN': []}

    for sample in image_stats_df.loc[('real', opt_thr, opt_overlap_thr, str(opt_scaling_factors))].iterrows():
        real_stats['TP'].extend(sample[1]['TP'])
        real_stats['FP'].extend(sample[1]['FP'])
        real_stats['TN'].extend(sample[1]['TN'])
        real_stats['FN'].extend(sample[1]['FN'])

    TP_count = pd.Series(real_stats['TP']).value_counts().reindex(classes, fill_value=0)
    FP_count = pd.Series(real_stats['FP']).value_counts().reindex(classes, fill_value=0)
    TN_count = pd.Series(real_stats['TN']).value_counts().reindex(classes, fill_value=0)
    FN_count = pd.Series(real_stats['FN']).value_counts().reindex(classes, fill_value=0)


    count_df = pd.concat([TP_count, FP_count, FN_count], axis=1)
    count_df.columns = ['TP', 'FP', 'FN']

    sns.set_context('paper')
    fig = plt.figure(figsize=(25,3))
    ax = plt.subplot(111)
    count_df.plot(ax=ax, kind='bar')
    ax.legend(loc='upper left')

    if logging:
        fig.savefig(os.path.join(logdir, 'figures/real_image_stats.png'), bbox_inches='tight')


    image_stats_df = pd.DataFrame(tuning_df)
    image_stats_df.scaling_factors = image_stats_df.scaling_factors.astype(str)

    image_stats_df = image_stats_df.set_index(['data_type', 'thr', 'overlap_thr', 'scaling_factors'])

    real_stats = {'TP': [], 'FP': [], 'TN': [], 'FN': []}

    for sample in image_stats_df.loc[('artificial', opt_thr, opt_overlap_thr, str(opt_scaling_factors))].iterrows():
        real_stats['TP'].extend(sample[1]['TP'])
        real_stats['FP'].extend(sample[1]['FP'])
        real_stats['TN'].extend(sample[1]['TN'])
        real_stats['FN'].extend(sample[1]['FN'])

    TP_count = pd.Series(real_stats['TP']).value_counts().reindex(classes, fill_value=0)
    FP_count = pd.Series(real_stats['FP']).value_counts().reindex(classes, fill_value=0)
    TN_count = pd.Series(real_stats['TN']).value_counts().reindex(classes, fill_value=0)
    FN_count = pd.Series(real_stats['FN']).value_counts().reindex(classes, fill_value=0)


    count_df = pd.concat([TP_count, FP_count, FN_count], axis=1)
    count_df.columns = ['TP', 'FP', 'FN']

    sns.set_context('paper')
    fig = plt.figure(figsize=(25,3))
    ax = plt.subplot(111)
    count_df.plot(ax=ax, kind='bar')
    ax.legend(loc='upper left')

    if logging:
        fig.savefig(os.path.join(logdir, 'figures/artificial_image_stats.png'), bbox_inches='tight')


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


############################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--json", type=str, default='',
                    help="Optional JSON file defining the training parameters")
args = parser.parse_args()

if __name__ == '__main__':

    json_file = str(args.json)

    # load parameter dictionary if specified as an argument
    if json_file:
        with open(json_file, 'r') as f:
            params = json.load(f)

    # otherwise define it manually
    else:
        # Specify folders from which to take data:
        data_directories = [
            'data/FIDS30',
            'data/original_clean',
            'data/update_9sep'
        ]

        # Folder for validation set
        real_validation_path = 'data/validation_real/'
        real_artificial_path = 'data/validation_artificial/'

        # define training parameters
        params = {
            ### Define experiment name:
            'experiment_name': 'test_all_again',

            ### Parameters for the CNN:
            'data_directories': data_directories,
            'test_size': 0.1,
            'seed': 11,
            'batch_size': 32,
            'target_size': (112,112),
            'epochs_cold': 1,
            'epochs_finetune': 1,
            'lr_cold': 0.001,
            'lr_finetune': 1e-5,

            'base_net': 'mobilenet_v2', # supported: resnet50/mobilenet_v2
            'head_net': '[tf.keras.layers.GlobalAveragePooling2D(),\
                         tf.keras.layers.Dropout(0.2)]',

            # The following parameters are for tuning the sliding window algorithm:
            'real_validation_path': real_validation_path,
            'artificial_validation_path': real_artificial_path,
            'thr_list': [0.9, 0.93, 0.96],
            'overlap_thr_list': [0.2, 0.3, 0.5],#list(np.arange(0,1,0.05)),
            'scaling_factors': [1.0, 1.5, 2.0],
            'sliding_strides': [32, 64, 128]
        }


    # execute main function
    main(params)
