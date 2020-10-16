import streamlit as st
import numpy as np
import cv2
import os
import sys
import inspect
import json

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import food_identification.models.model_helpers as model_helpers
from recipes.similarity_finder_for_app import find_similar_recipes

# Importing the inception_v3 as the preprocessing model
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3

# Getting the model
from tensorflow.keras.models import Sequential, save_model, load_model

from PIL import Image

# setting style
with open("./utils/style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# load model parameters
model_path = '../food_identification/models/final_model/'
with open(os.path.join(model_path, 'results_classifier.json')) as fp:
    results_classifier = json.load(fp)

classes = results_classifier['classes']
ind2class = results_classifier['ind2class']
ind2class = {int(k): v for (k, v) in ind2class.items()}

# Defining the parameters for the sliding window model
preprocess_func = preprocess_inception_v3

# define correct input size for the network (the one it was trained on)
kernel_size = 224

# define selection threshold / do not take prediction with a lesser confidence level
thr = 0.95

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


# WEB APP
# Actual app

st.text("")
st.text("")
st.text("")
st.image('./utils/output-onlinepngtools.png')


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

st.write("### Upload your ingredients, deepfoodie is waiting to prepare some recipes for you!")


""
""
"We will begin by analyzing the ingredients that are available to you"

""
""

"Please, upload a picture so that deepfoodie can process it 🤖"

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
    pred_labels, probabilities, x0, y0, windowsize = make_prediction(img)
    
    # initialize function of detection
    
    st.write('This is what deep-foodie has found')
    fig = model_helpers.visualize_predictions(
                                        img,
                                        pred_labels,
                                        probabilities,
                                        x0,
                                        y0,
                                        windowsize
                                              )
    st.write(fig)
    
    st.write("Okay, here are the ingredients that you have:")
    ""
    pred_labels = np.unique(pred_labels)
    st.write(pred_labels)
    st.write("Would you like to add something more?")
    
    np.save("../recipes/input/pred_labels.npy", pred_labels, allow_pickle=True) 

if st.button('show me recipes'):
    pred_labels = np.load("../recipes/input/pred_labels.npy", allow_pickle=True)
    pred_labels_l = list(pred_labels)
    recipes = find_similar_recipes(pred_labels_l)  

    for index, row in recipes.iterrows():
        st.title(row['title'])
        st.write(row['url'])
        st.subheader('Ingredients:')
        st.write(row['ingredients']) 
    
    

