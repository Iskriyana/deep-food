FROM python:3.6.10

RUN pip install virtualenv
ENV VIRTUAL_ENV=/deep-food
RUN virtualenv deep-food -p python3.6.10
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /deepfoodie
ADD . /deepfoodie

# Expose port 
ENV PORT 8081

# Run the application:
CMD ["gunicorn", "app:deepfoodie", "--config=config.py"]