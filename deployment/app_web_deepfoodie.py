"""Deepfoodie main app
"""

__author__ = "Iskriyana Vasileva, Gleb Sidorov, Nima H. Siboni, Michael Drews"

import streamlit as st
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.path.append('..')
import food_identification.models.model_helpers as model_helpers
from recipes.similarity_finder_for_app import find_similar_recipes
from data.data_helpers import resize_image_to_1024
from config import *
import app_utils as app_utils


# setting style
with open("./utils/style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


def main():
    """Main web app
    """

    # Write header
    st.text("")
    st.text("")
    st.text("")
    st.image('./utils/output-onlinepngtools.png')

    st.write("""
    # \t 🍏 **Deep Food** 🍌
    """)
    st.text("")
    st.text("")

    st.write("""
    Final project of Data Science Retreat, brought to you in collaboration by:
    - Michael Drews
    - Nima Siboni
    - Gleb Sidorov
    - Iskriyana Vasileva
    """)

    st.write("### Upload your ingredients, deepfodie is waiting to prepare some recipes for you!")


    ""
    ""
    "We will begin by analyzing the ingredients that are available to you"

    ""
    ""

    "Please, upload a picture so that deepfoodie can process it 🤖"
    
    # generate file dialog
    st.set_option('deprecation.showfileUploaderEncoding', False)
    file = st.file_uploader("Upload file", type=["jpg"])
    
    if file is None:
        st.write("""Or use our example picture:""")
        file = './utils/file46.jpg'

    if file:
        # after file upload show image
        img = np.array(Image.open(file))
        img = resize_image_to_1024(img)

        fig = plt.figure(figsize=(12,12))
        ax = plt.subplot(111)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        st.write(fig)

        # show activate button
        activate_vision = st.button("deepfoodie, activate your vision!")
        #st.write(activate_vision)

        if activate_vision:
            # after button click make predictions
            st.write("Request accepted")

            # run model
            pred_labels, probabilities, x0, y0, windowsize  = app_utils.make_prediction(
                img, 
                MODEL_SPECS, 
                IND2CLASS,
                sliding_strides=[64] 
            )
            
            # show celebratory balloons
            if len(pred_labels)>0:
                st.balloons()

            # show results
            st.write('This is what deepfoodie has found:')
            st.write('')
            fig = model_helpers.visualize_predictions(
                img, 
                pred_labels, 
                probabilities, 
                x0, 
                y0,
                windowsize)
            st.write(fig)

            st.write("Okay, here are the ingredients that you have:")
            pred_labels = np.unique(pred_labels)
            nl = '\n'
            st.text(nl.join(list(pred_labels)))
            np.save("../recipes/input/pred_labels.npy", pred_labels, allow_pickle=True) 
            
            st.write("Please wait a few seconds until deepfoodie has scanned our recipe database...")
            
            #st.write("Would you like to add something more?")

            #st.write("#### Would you like to know what you can cook with this?")
            #if st.button('deepfoodie, show me recipes!'):

            st.write('#### Here are some suggestions for what you can prepare with this:')
            st.write('')
            pred_labels = np.load("../recipes/input/pred_labels.npy", allow_pickle=True)
            pred_labels_l = list(pred_labels)
            recipes = find_similar_recipes(pred_labels_l)  #

            for index, row in recipes.iterrows():
                st.title(row['title'])
                st.write(row['url'])
                st.subheader('Ingredients:')
                st.write(row['ingredients']) 

    
if __name__ == "__main__":
    main()