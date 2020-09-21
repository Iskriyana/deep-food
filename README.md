# deep-food

<img src="https://github.com/Iskriyana/deep-food/blob/master/assets/deep_food.jpg" width=300/>

## Dockerfile

The Docker file image of this project can be found at:

`https://hub.docker.com/repository/docker/gsrus96/deep-food`

In order to run the docker container programmatically from your PC,
Pull the docker image from the Docker hub and then mount the image and run it:
`docker run -it -w="/deep-food" [Docker_Image_ID] streamlit run app_web.py`

### Installation

`conda create -n deep_food python=3.6`

`conda activate deep_food`

`pip install -r requirements.txt`

### Generate new artificial validation set

For 500 samples, use:

`python generate_artifical_validation_set.py validation_artificial --N_samples 500`
