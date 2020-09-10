"""Some helper functions to handle the data flow."""

import os, glob
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import cv2


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

def get_train_test_data_df(data_directories, test_size, seed):
    """
    Standard pipeline to get train and test dataframes.

    Args:
        data_directories: directories to search for images
        test_size: size of test set
        seed: random state for train-test split

    Returns:
        data_df_train: train set DataFrame
        data_df_test: test set DataFrame
        classes: list of n_classes
        class2ind: dict-mapping from classes to indices
        ind2class: dict-mapping from indices to classes
    """
    # get base data frame
    data_df = generate_data_df(data_directories)
    classes = list(sorted(data_df.label.unique()))
    n_classes = len(classes)

    class2ind = {c:i for (i, c) in enumerate(classes)}
    ind2class = {i:c for (c,i) in class2ind.items()}

    # train test split
    data_df_train, data_df_test = train_test_split(data_df, test_size=test_size,
                                                   stratify=data_df.label, random_state=seed)

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


def get_validation_dict(path, classes, verbose=0):
    """
    Generates a dict containing images and classes for the validation set

    Args:
        path: Location of the validation set
        classes: List of registered classes in the training set

    Returns:
        val_data: dict containting images and classes
    """

    validation_imgfiles = glob.glob(os.path.join(path, '*.jpg'))
    validation_textfiles = [os.path.splitext(vf)[0]+'.txt' for vf in validation_imgfiles]

    val_data = {}

    for i, (img_file, txt_file) in enumerate(zip(validation_imgfiles, validation_textfiles)):
        with open(txt_file) as f:
            txt = f.read()
        real_classes = txt.split('\n')

        # get classes which exist in our dataset
        take_classes = set(real_classes).intersection(set(classes))

        # print classe which are discarded because they do not exist in dataset
        discard_classes = set(real_classes) - set(classes)
        if verbose >= 1:
            print(f'File: {txt_file}')
            print(f'Discarded: {discard_classes}')
            print()

        # register classes in dict
        val_data[i] = {'image': img_file, 'classes': take_classes}

    return val_data


def load_bg_random(path):
    """
    Load a random background image from bg_folder

    Args:
        path: folder to load images flow_from_dataframe

    Returns:
        background: image
        rows_b: number of rows
        cols_b: number of columns
        channels_b: number of channels
    """
    bg_folder = path
    bg_files = os.listdir(bg_folder)
    bg_files = [f for f in bg_files if not f[0] == '.']
    #bg_index = random.randrange(0, len(bg_files))
    bg_index = np.random.randint(0, len(bg_files))

    bg = os.path.join(path, bg_files[bg_index])

    background = cv2.imread(bg)
    rows_b, cols_b, channels_b = background.shape

    return background, rows_b, cols_b, channels_b


def assemble_img_grid_from_df(data_df, bg_path, ingr_dict, ingr_size, grid_step, size_jitter=(1.0,1.0)):
    """
    Takes in a dictionary of ingredients and their quantities.
    Randomly picks a number of images from the specified ingredient category.
    Randomly picks a background image.
    Creates a grid on the background image.
    Pastes the ingredient images on the background image.

    Args:
        data_df: DataFrame with filepaths and labels
        bg_path: path to folder containing background images
        ingr_dict: dict indicating number of elements per class
        ingr_size: (tuple) size of element
        grid_step: spacing between rows and columns

    Returns:
        bg: artificially generated image
        ingredients: list of pasted images
        coords: coordinates of images
    """

    # list with ingredient images
    ingredients = []
    labels = list(ingr_dict.keys())
    n_ingr = sum(ingr_dict.values())

    for label in labels:
        ingr_files = data_df.set_index('label').loc[label].values

        sample_size = ingr_dict[label]
        #random choice of image with repetition
        #ingredient_basket = random.choices(ingr_files, k=sample_size)
        ingredient_basket = [np.random.choice(ingr_files.flatten(), size=sample_size)]
        #import pdb; pdb.set_trace()
        rows, cols = ingr_size
        rows = int(np.random.uniform(size_jitter[0], size_jitter[1])*rows)
        cols = int(np.random.uniform(size_jitter[0], size_jitter[1])*cols)

        for ing in ingredient_basket:
            #ingredient_path = os.path.join(path, ingr_folder, ing)
            ingredient = Image.open(ing[0])
            ingredient = ingredient.resize((rows, cols))
            ingredients.append(ingredient)

    # background generation
    background = load_bg_random(bg_path)[0]
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    background = Image.fromarray(background)
    bg = background.resize((512, 1024)).copy()
    rows_b, cols_b = bg.size

    # grid definition
    grid_width = bg.size[0]
    grid_height = bg.size[1]
    grid_step = grid_step
    x = np.arange(0, grid_width - ingr_size[0], grid_step)
    y = np.arange(0, grid_height - ingr_size[1], grid_step)
    X,Y = np.meshgrid(x,y)
    coords = np.array(list(zip(X.flatten(), Y.flatten())))

    #no replacement, because otherwise the images get overwritten
    coords = coords[np.random.choice(np.arange(len(coords)), size = n_ingr, replace = False), :]

    # image pasting
    for i, ingredient in enumerate(ingredients):
        bg.paste(ingredient, (coords[i][0], coords[i][1]))

    #print(list(zip(labels, coords)))
    return bg, ingredients, coords


def gen_one_artifical_image(data_df, classes, bg_path, N_min=5, N_max=10, spacing=150, size_jitter=(1.0,1.0)):
    """Generate one artificial sample.

    Args:
        data_df: DataFrame with filepaths and labels
        classes: class_labels
        bg_path: path to background images
        N_min: minimum number of ingredients
        N_max: maximum number of ingredients
        spacing: pixel spacing between ingredients
        size_jitter: (min, max) range for random scale factor
        seed: seed for random generator

    Returns:
        img: artifical image
        labels: list of labels
    """

    N = np.random.randint(N_min, N_max+1)
    size = (spacing, spacing)

    ingredients = [classes[i] for i in np.random.randint(0, len(classes), N)]
    ingredients = pd.Series(ingredients).value_counts().to_dict()

    img, _, _ = assemble_img_grid_from_df(data_df,
                                          bg_path,
                                          ingredients,
                                          tuple(size),
                                          spacing,
                                          size_jitter=size_jitter)

    labels = set(ingredients.keys())

    return img, labels
