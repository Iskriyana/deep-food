FROM python:3.6.10

RUN pip install virtualenv
ENV VIRTUAL_ENV=/deep-food
RUN virtualenv deep-food -p python3.6.10
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port 
ENV PORT 8080

# Run the application:
CMD ["gunicorn", "app:app_web_deepfoodie", "--config=config.py"]