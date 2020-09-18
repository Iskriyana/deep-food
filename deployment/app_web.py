import streamlit as st
import os
import time


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
st.image('deep_foodie.png')


""
""
"We will begin by analyzing the ingredients that are available to you"

""
""

"Please, upload a picture so we can process it so deep-foodie can process it 🤖"

st.set_option('deprecation.showfileUploaderEncoding', False)
file = st.file_uploader("Upload file", type=["png"])

if file:
    st.image(file)

activate_vision = st.button("deep-foodie activate your vision")

print(activate_vision)


if activate_vision:
    st.write("Request accepted")
    
    #initialize function of detection
    
    time.sleep(2)
    st.write("Let's see what you've got...")
    time.sleep(2)
    st.write("ahaaa... not the cleanliest fridge, noted")
    time.sleep(2)
    st.write("I am almost done, don't be pushy")
    time.sleep(2)
    st.write("Analysis completed ✔️")
    
