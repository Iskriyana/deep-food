FROM python:3.6.10

RUN pip install virtualenv

RUN virtualenv deep-food -p python3.6.10
RUN export PATH=/usr/local/bin:$PATH

WORKDIR /deep-food
ADD . /deep-food

# Install dependencies
RUN pip3 install -r requirements_slim.txt

RUN apt-get update && apt-get install -y libgl1-mesa-dev

WORKDIR /deep-food/deployment
# Expose port 
EXPOSE 8501


# Run the application:
CMD ["streamlit","run","./app_web_deepfoodie.py"]