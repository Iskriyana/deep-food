# deep-food

<img src="https://github.com/Iskriyana/deep-food/blob/master/assets/deep_food.jpg" width=300/>

## Project Intro
* Have you ever wished you can make a photo of your fridge, upload it and get a recipe recommendation?
* This is what we attempt with our app deepfoodie: personalised recipe generation based on ingredients recognition from a photo

## Collaborators
|Name               |  Github Page                    |  
|-------------------|---------------------------------|
|Michael Drews      | (https://github.com/michi-d)    |
|Nima H. Siboni     | (https://github.com/nima-siboni)|
|Iskriyana Vasileva | (https://github.com/Iskriyana)  |
|Gleb Sidorov       | (https://github.com/gsidorov)   |

## Methods Used & Tech Stack 
* Scraping - selenium, requests
* Image manipulation & artificial image creation - Pillow, OpenCV
* Deep Learning - Convolutional NN with TensorFlow
* Transfer Learning - Resnet50, MobileNetV2, InceptionV3 

## Getting Started
* `git clone https://github.com/Iskriyana/deep-food.git`

* `conda create -n deep-food python=3.6.10`

* `conda activate deep-food`

* `pip install -r requirements.txt`

* Download the following files in the folders as listed below:
    * in data/recipe_data_sets/output
        * https://drive.google.com/file/d/1BRKp-h3b8-KoyesG1Q39f15cZFPUajRx/view?usp=sharing
        * https://drive.google.com/file/d/1hAk4KfT0fDOLRWXiK7WLi0oH1whhuA7w/view?usp=sharing
    * in recipes/input - https://drive.google.com/file/d/1-5HUvh4ho3BLdZo3LRnOsrz4CeVp24jj/view?usp=sharing 
    * in models/final_model/variables - https://drive.google.com/file/d/1WLNszn9oGpckj08JaxIp3aF2xLvTL5XJ/view?usp=sharing

## Link to App 
http://34.75.113.48/

## If you want to run the web app locally
* Go to the deployment folder `cd deployment`
* Start the application `streamlit run app_web_deepfoodie.py`
  * In case you get the error "ModuleNotFoundError: No module named 'importlib_metadata'", deactivate and activate the environment again
* Upload a photo of your fridge or use our example photo in /assets/example_photo.jpg
* Click on "deep-foodie activate your vision". The model is now running in order to identify the ingredients in the image.
* Once deepfoodie is ready with the ingredients identification, you will see your picture with red squares and labels of what deepfoodie thinks he saw
* Now, click on "show me recipes" to get 3 gourmet recommendations. Enjoy!


## Other useful points

### Generate new artificial validation set
For 500 samples, use:

`python /data/generate_artifical_validation_set.py validation_artificial --N_samples 500`

