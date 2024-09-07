# Celebrity-Look-Alike

# Overview

This project aims to extract face embeddings from celebrity images using a pre-trained deep learning model. Face embeddings are compact representations of facial features, which can be used for various applications, including face recognition, similarity analysis, and clustering. The project leverages a modified ResNet model to produce 128-dimensional embeddings from facial images.

# Features

## Face Detection: Detects faces in images using a face detection model.

## Face Embeddings Extraction: Utilizes a modified ResNet model to extract 128-dimensional embeddings for each detected face.

## Model Fine-Tuning: Includes steps for fine-tuning the model on a dataset of celebrity images to improve embedding quality.

## Data Handling: Demonstrates loading and handling embeddings stored in a .pkl file.

## Deployment: Provides a Streamlit app for interactive face embedding extraction and visualization.

# Requirements

Python 3.x

PyTorch

torchvision

OpenCV

Pillow

Streamlit

A pre-trained face detection model (e.g., MTCNN or Haar cascades)
