import streamlit as st
import os
import time
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import cv2
import glob, os
import json

from PIL import Image
# TRYING TO IMPORT ALL THE DIFFICULT THINGS

# import data.data_helpers as data_helpers
import models.model_helpers as model_helpers
# import models.tuning_helpers as tuning_helpers

## setting style test!
with open("./deployment/style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


# import sliding_window_dep
# I DO NOT KNOW HOW TO MAKE THIS WORK!!


# Importing the inception_v3 as the preprocessing model
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3

# load model parameters
model_path = './models/final_model/'
with open(os.path.join(model_path, 'results_classifier.json')) as fp:
    results_classifier = json.load(fp)

classes = results_classifier['classes']
ind2class = results_classifier['ind2class']
ind2class = {int(k):v for (k,v) in ind2class.items()}

# Getting the model

from tensorflow.keras.models import Sequential, save_model, load_model



#THIS PART MAKES THE WEB APP SUPER SLOW



# loaded_model = load_model(
#     model_path,
#     custom_objects=None,
#     compile=True
# )




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

@st.cache
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











# WEB APP

# Actual app

st.write("""
#             \t 🍏 **Deep Food** 🍌""")

st.text("")

st.text("")

st.write("""
Final project of Data Science Retreat, brought to you in collaboration by:
- Michael Drews
- Nima Siboni
- Gleb Sidorov
- Iskriyana Vasileva
""")

st.write("### Upload your ingredients, our AI robot is waiting to prepare you some recipes for you!")


st.text("")
st.text("")
st.text("")
st.image('./deployment/output-onlinepngtools.png')


""
""
"We will begin by analyzing the ingredients that are available to you"

""
""

"Please, upload a picture so we can process it so deep-foodie can process it 🤖"

st.set_option('deprecation.showfileUploaderEncoding', False)
file = st.file_uploader("Upload file", type=["jpg"])

if file:
    img = np.array(Image.open(file))
    scaling_factor = max(img.shape)/1024.
    new_shape = (int(img.shape[1]/scaling_factor), int(img.shape[0]/scaling_factor))
    img = cv2.resize(img, new_shape)
    st.image(img)

activate_vision = st.button("deep-foodie activate your vision")

print(activate_vision)

pred_labels = np.array([])
probabilities = np.array([])
x0 = np.array([])
y0 = np.array([])
windowsize = np.array([])

if activate_vision:
    st.write("Request accepted")
    
    st.write("Loading Model...")
    loaded_model = load_model(
    model_path,
    custom_objects=None,
    compile=True
)
    st.write('Making Predictions...')
    pred_labels, probabilities, x0, y0, windowsize  = make_prediction(img)
    
    #initialize function of detection
    
    st.write('This is what deep-foodie has found')
    fig = model_helpers.visualize_predictions(img, 
                                          pred_labels, 
                                          probabilities, 
                                          x0, 
                                          y0,
                                          windowsize)
    st.write(fig)
    
    st.write("Okay, here are the ingredients that you have:")
    ""
    pred_labels = np.unique(pred_labels)
    st.write(pred_labels)
    st.write("Would you like to add something more?")
    
    np.savetxt("ingredients.csv", pred_labels, delimiter=",", encoding='utf-8', fmt='%s')
    

if st.button('show me recipes'):
    recipes = pd.read_csv('recipes.csv', delimiter = ';', encoding = "ISO-8859-1")
    
    recipe_name = recipes.loc[:, 'Name']
    recipes_ingr = recipes.loc[:, 'Ingredients']
    recipes_instructions = recipes.loc[:, 'Instructions']
    recipe_url = recipes.loc[:, 'URL']    
    
    #recipes_dict = recipes.to_dict()
    #st.write(recipes_dict)
    for recipe in range(recipes.shape[0]):
        st.title(recipe_name[recipe])
        st.write(recipe_url[recipe])
        st.subheader('Ingredients:')
        st.write(recipes_ingr[recipe]) 
        st.subheader('Instructions:')
        st.write(recipes_instructions[recipe])
        #st.write(recipes.iloc[recipe, :])
    #st.dataframe(recipes)


    
    

