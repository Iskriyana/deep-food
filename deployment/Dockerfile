FROM python:3.6.10

RUN pip install virtualenv
ENV VIRTUAL_ENV=/deep-food
RUN virtualenv deep-food -p python3.6.10
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /deepfoodie
ADD . /deepfoodie

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose port 
ENV PORT 8081

# Run the application:
CMD ["streamlit", "app:app_web_deepfoodie.py", "--config=config.py"]