"""Some helper functions to handle the data flow."""

import os, glob
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def is_image(filename):
    """Checks if input is an image file"""
    _, ext = os.path.splitext(filename)
    return ext in {'.png', '.jpg', '.jpeg'}


def clean_label(label):
  """
  Standardize class names (e.g. singular/plural versions)
  """

  if label == 'apples':
    label = 'apple'
  elif label == 'avocados':
    label = 'avocado'
  elif label == 'bananas':
    label = 'banana'
  elif label == 'cantaloupes':
    label = 'melon'
  elif label == 'kiwifruit':
    label = 'kiwi'
  elif label == 'lemons':
    label = 'lemon'
  elif label == 'limes':
    label = 'lime'
  elif label == 'mangos':
    label = 'mango'
  elif label == 'oranges':
    label = 'orange'
  elif label == 'pineapples':
    label = 'pineapple'
  elif label == 'strawberries':
    label = 'strawberry'
  elif label == 'watermelons':
    label = 'watermelon'
  elif label == 'blackberries':
    label = 'blackberry'
  elif label == 'blueberries':
    label = 'blueberry'
  elif label == 'olives':
    label = 'olive'
  elif label == 'peaches':
    label = 'peach'
  elif label == 'pears':
    label = 'pear'
  elif label == 'plums':
    label = 'plum'
  elif label == 'pomegranates':
    label = 'pomegranate'
  elif label == 'raspberries':
    label = 'raspberry'
  elif label == 'tomatoes':
    label = 'tomato'

  return label


def generate_data_df(data_directories):
    """
    Generate base data frame containing all image file paths.

    Args:
        data_directories: list of paths to include
    """

    filepaths = []
    labels = []
    for folder in data_directories:
        # get list of all files in all subdirectories of each folder
        all_files = glob.glob(os.path.join(folder, '*', '*'))

        for path in all_files:
            if is_image(path):
                filepaths.append(path)

                parent_folder, _ = os.path.split(path)
                _, folder = os.path.split(parent_folder)
                labels.append(folder)

        # generate pandas DataFrame
    data_df = pd.DataFrame({'file': filepaths, 'label': labels})
    data_df.label = data_df.label.apply(clean_label)

    return data_df

def get_train_test_data_df(params):
    """
    Standard pipeline to get train and test dataframes.

    Args:
        params: parameter dictionary for this model

    Returns:
        data_df_train: train set DataFrame
        data_df_test: test set DataFrame
        classes: list of n_classes
        class2ind: dict-mapping from classes to indices
        ind2class: dict-mapping from indices to classes
    """
    # get base data frame
    data_df = generate_data_df(params['data_directories'])
    classes = list(sorted(data_df.label.unique()))
    n_classes = len(classes)

    class2ind = {c:i for (i, c) in enumerate(classes)}
    ind2class = {i:c for (c,i) in class2ind.items()}

    # train test split
    data_df_train, data_df_test = train_test_split(data_df, test_size=params['test_size'],
                                                   stratify=data_df.label, random_state=params['seed'])

    return data_df_train, data_df_test, classes, class2ind, ind2class


def get_data_frame_iterator(df, image_generator, params):
    """Generate standard keras DataFrameIterator from dataframe and ImageGenerator

    Args:
        df: DataFrame containing files and labels
        image_generator: keras.ImageDataGenerator object

    Returns:
        iterator: keras.DataFrameIterator object for this data
    """
    iterator = image_generator.flow_from_dataframe(dataframe=df,
                                                   x_col='file',
                                                   y_col='label',
                                                   batch_size=params['batch_size'],
                                                   shuffle=True,
                                                   target_size=params['target_size'],
                                                   seed=params['seed'])
    return iterator


def deprocess_img(processed_img):
    """
    Function to invert the ImageNet-preprocessing.
    (Source: https://stackoverflow.com/questions/55987302/reversing-the-image-preprocessing-of-vgg-in-keras-to-return-original-image)
    """
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')

    return x


def image_grid(image_batch, label_batch, ind2class, n_row=2, n_col=5, deprocess=True):
    """
    Generate figure with example pictures on a grid.

    Args:
        image_batch: (batch_size x width x height x 3) array
        label_batch: (batch_size x n_classes) one-hot encoded array
        ind2class: dict-mapping from indices to class names

    Returns:
        figure: matplotlib.Figure containing the plot
    """

    N = n_row * n_col
    figure = plt.figure(figsize=(n_col*2, n_row*2))
    for i in range(N):
        plt.subplot(n_row, n_col, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        if deprocess:
            img = deprocess_img(image_batch[i])
        else:
            img = image_batch[i]
        plt.imshow(img)

        ind = np.nonzero(label_batch[i])[0][0]
        plt.title(ind2class[ind], color='r')

    return


def show_label_distribution(data_df, ind2class, title=''):
    """
    Plot label count histogram for given dataset

    Args:
        data_df: base dataframe with for a given dataset
        ind2class: dict-mapping from indices to class names
        title: optional title for the plot

    Returns:
        figure: matplotlib.Figure containing the plot
    """

    labels = pd.Series(data_df.label).replace(ind2class)

    fig = plt.figure(figsize=(20,5))
    ax = plt.subplot(111)

    labels.value_counts().sort_index(0).plot(kind='bar')
    plt.xticks(color='r', rotation=90, fontsize=15)
    plt.yticks(color='r', fontsize=15)
    plt.title(title, color='red', fontsize=15)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()))

    return fig
