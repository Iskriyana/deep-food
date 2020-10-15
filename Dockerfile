FROM python:3.6.10

RUN pip install virtualenv
#RUN python -m virtualenv deep-food -p python3.6.10
#RUN python source deep-food/bin/activate

#ENV VIRTUAL_ENV=/deep-food
RUN virtualenv deep-food -p python3.6.10
#ENV PATH="VIRTUAL_ENV/bin:$PATH"
RUN export PATH=/usr/local/bin:$PATH

WORKDIR /deep-food
ADD . /deep-food
# Install dependencies
RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-dev

WORKDIR /deep-food/data/recipes_data_set/output
# Add recipe files: recipes_clean & recipes_1M_shortened.csv
RUN gdown https://drive.google.com/uc?id=1BRKp-h3b8-KoyesG1Q39f15cZFPUajRx
RUN gdown https://drive.google.com/uc?id=1hAk4KfT0fDOLRWXiK7WLi0oH1whhuA7w

WORKDIR /deep-food/recipes/input
# Add one-hot encoded recipes: recipes_one_hotted.npy
RUN gdown https://drive.google.com/uc?id=1-5HUvh4ho3BLdZo3LRnOsrz4CeVp24jj

WORKDIR /deep-food/food_identification/models/final_model/variables
# Add variables.data-00001-of-00002 file
RUN gdown https://drive.google.com/uc?id=1WLNszn9oGpckj08JaxIp3aF2xLvTL5XJ

WORKDIR /deep-food/deployment
# Expose port 
EXPOSE 8501


# Run the application:
CMD ["streamlit","run", "./app_web_deepfoodie.py"]
# ENTRYPOINT streamlit run --server.port 8501 --server.enableCORS false app_web_deepfoodie.py