# MLOps-Animal-Classifier

## Dataset
https://www.kaggle.com/datasets/alessiocorrado99/animals10

For the Dockerfile:

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (optional, for Flask)
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
