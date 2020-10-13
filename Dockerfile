FROM python:3.6.10

#RUN pip install virtualenv
#ENV VIRTUAL_ENV=/deep-food
#RUN virtualenv deep-food -p python3.6.10
#ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /deep-food
ADD . /deep-food
# Install dependencies
RUN pip3 install -r requirements.txt

WORKDIR /deep-food/data/recipes_data_set/output
# Add recipe files: recipes_clean & recipes_1M_shortened.csv
ADD https://drive.google.com/file/d/1BRKp-h3b8-KoyesG1Q39f15cZFPUajRx/view?usp=sharing \
/deep-food/data/recipes_data_set/output

ADD https://drive.google.com/file/d/1hAk4KfT0fDOLRWXiK7WLi0oH1whhuA7w/view?usp=sharing \
/deep-food/data/recipes_data_set/output

WORKDIR /deep-food/recipes/input
# Add one-hot encoded recipes: recipes_one_hotted.npy
ADD https://drive.google.com/file/d/1-5HUvh4ho3BLdZo3LRnOsrz4CeVp24jj/view?usp=sharing \
/deep-food/recipes/input

WORKDIR /deep-food/deployment
# Expose port 
EXPOSE NV PORT 8501

# Run the application:
CMD ["streamlit", "./app_web_deepfoodie.py"]
# ENTRYPOINT streamlit run --server.port 8501 --server.enableCORS false app_web_deepfoodie.py