# MLOps-Animal-Classifier

[Visit the Website Here](https://yckamra.github.io/MLOps-Animal-Classifier/root/)

## Overview
  This project is a basic image classifier using PyTorch's ResNet32 CNN trained on the ImageNet 1k dataset, and fine-tuned on the [Animal-10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10) from Kaggle, and served via an API using FastAPI. The pipeline was end-to-end, including:
  * EDA and cleaning of data with Matplotlib and pandas
  * Training/fine-tuning the hyperparameters of the CNN
  * Logging each experiment with MLFlow
  * Creating API methods using FastAPI
  * GitHub Actions for CI/CD
  * Google Cloud Platform's Cloud Storage, Artifact Registry, and Cloud Run
  * Basic front-end website to demo the APIs with Javascript, HTML, and CSS
    
The website is able to classify images of butterflies, cats, chickens, cows, dogs, elephants, horses, sheep, spiders, and squirrels.

## Training

The Animal-10 dataset was split into training, cross-validation, and test sets with the split 70-15-15 respectively. The following hyperparameters were found to produce the best metrics on the cross-validation set, but random search or grid search could automate the process of finding better values:
  * Epochs = 10
  * Learning Rate = 0.001
  * Batch Size = 64
  * Momentum = 0.4
  * Weight Decay = 1e-4

On the cross-validation set, the model achieved a 90.12% accuracy and an F-1 Score of 0.9. While these metrics are not outstanding, the Animal-10 dataset contains many images of animals in the distance, poor lighting, multiple entities within the image. Further hyperparameter exploration could also help improve the model. However, for the sake of learning MLOps and exploring the process of end-to-end projects and as the API website service is not critical to human safety (ie. health care predictions where model precision and recall must also be heavily understood), the metrics are fine for the purpose of this project; I'll probably improve these at a later date.

## How to Use the Website

The website is easy to use. Take any picture of the ten accepted animals, put it in .jpg or .jpeg format, click the "Chose File" and select the image, and then click "Predict". The first prediction may take ~10-20 seconds as Google Cloud Run must wake up as it is set to become dormant after a set period of time. It is pretty fun to see if your family cat, dog, or chicken is accurately classified.

## What I Learned

Building an end-to-end MLOps pipeline showed me a lot about the Google Cloud Platform, EDA, MLFlow, Docker, FastAPI, and GitHub Actions. This is just the beginning of learning these libraries, frameworks, and processes so I am extremely excited to explore further into each of their capabilities.
