import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import cv2
import glob, os, sys, inspect
import json

import models.tuning_helpers as tuning_helpers
import models.model_helpers as model_helpers
from PIL import Image

from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import data.data_helpers as data_helpers

# Importing the inception_v3 as the preprocessing model
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3

# load model parameters
with open(os.path.join(model_path, 'results_classifier.json')) as fp:
    results_classifier = json.load(fp)

classes = results_classifier['classes']
ind2class = results_classifier['ind2class']
ind2class = {int(k):v for (k,v) in ind2class.items()}

# Getting the model

from tensorflow.keras.models import Sequential, save_model, load_model
model_path = 'models/final_model/'

loaded_model = load_model(
    model_path,
    custom_objects=None,
    compile=True
)

# Defining the parameters for the sliding window model
preprocess_func = preprocess_inception_v3

# define correct input size for the network (the one it was trained on)
kernel_size = 224

# define selection threshold / do not take prediction with a lesser confidence level
thr = 0.87

# define non-max suppression threshold
overlap_thr = 0.2

# define image pyramid (object sizes / larger factors correspond to smaller objects)
scaling_factors = [1.5]
sliding_strides = [64]

def make_prediction(image):
    pred_labels, probabilities, x0, y0, windowsize = \
    model_helpers.object_detection_sliding_window(model=loaded_model, 
                                                      input_img=image, 
                                                      preprocess_function=preprocess_func, 
                                                      kernel_size=kernel_size, 
                                                      ind2class=ind2class, 
                                                      scaling_factors=scaling_factors, 
                                                      sliding_strides=sliding_strides, 
                                                      thr=thr, 
                                                      overlap_thr=overlap_thr)
    return pred_labels, probabilities, x0, y0, windowsize

if __name__ == "__main__":
    # execute only if run as a script
    st.write('I HAVE BEEN IMPORTED')
    sliding_window_dep()

