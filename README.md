# deep-food
## Project Intro/Objective
* have you ever wished you can make a photo of your fridge, upload it and get a recipe recommendation?
* this is what we attempt with our app deepfoodie: personalised recipe generation based on ingredients recognition from a photo

## Collaborators
|Name     |  Github Page     |  Personal Website  |
|-------------------|---------------|---------------------------------|
|Michael Drews      | [michi-d]     | (https://github.com/michi-d)    |
|Nima H. Siboni     | [nima-siboni] | (https://github.com/nima-siboni)|
|Iskriyana Vasileva | [iskriyana]   | (https://github.com/Iskriyana)  |
|Gleb Sidorov       | [gsidorov]    | (https://github.com/gsidorov)   |

<img src="https://github.com/Iskriyana/deep-food/blob/master/assets/deep_food.jpg" width=300/>

## Methods Used & Tech Stack 
* Scraping - selenium, requests
* Image manipulation & artificial image creation - Pillow, OpenCV
* Deep Learning - Convolutional NN with TensorFlow
* Transfer Learning - Resnet50, MobileNetV2, InceptionV3 

## Getting Started

`conda create -n deep_food python=3.6.10`

`conda activate deep_food`

`pip3 install -r requirements.txt`

## Run Web App locally
* `cd deployment`
* `streamlit run app_web_deepfoodie.py`
* upload a photo of your fridge or use our example photo in /assets/example_photo.jpg
* click on "deep-foodie activate your vision". The model is now running in order to identify the ingredients in the image.
* once deepfoodie is ready with the ingredients identification, you will see your picture with red squares and labels of what deepfoodie thinks he saw
* now, click on "show me recipes" to get 3 gourmet recommendations. Enjoy!


## Other useful points

### Generate new artificial validation set
For 500 samples, use:

`python /data/generate_artifical_validation_set.py validation_artificial --N_samples 500`

