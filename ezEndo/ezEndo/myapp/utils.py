import io
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import os
from PIL import Image
from torchvision import models, transforms
from keras.models import load_model, Model
from django.conf import settings
from google.cloud import storage
import cv2

# Replace the paths to the actual paths on your device
classifier_model_path = 'C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/best_classifier.pth'
autoencoder_model_path = 'C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/autoencoder_basic.keras'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset(csv_file_path):
    data = pd.read_csv(csv_file_path)
    return data['image_path'].tolist(), data['label'].tolist()

def load_pytorch_classifier(model_path):
    model = models.alexnet(pretrained=False) 
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 2) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_keras_autoencoder(model_path):
    return load_model(model_path)

classifier = load_pytorch_classifier(classifier_model_path)
autoencoder = load_keras_autoencoder(autoencoder_model_path)

def preprocess_image_for_classifier(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0).to(device)

def preprocess_image_for_autoencoder(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)
    return np.expand_dims(image, axis=0)

def classify_image(image_bytes):
    tensor_image = preprocess_image_for_classifier(image_bytes)
    with torch.no_grad():
        outputs = classifier(tensor_image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    confidence_scores, predicted_classes = torch.max(probs, 1)
    label = 'Normal' if predicted_classes.item() == 0 else 'Abnormal'
    return label, confidence_scores.item()

def get_embedding(image_bytes):
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)
    images = preprocess_image_for_autoencoder(image_bytes)
    images = images.reshape(-1, 256, 256, 1)
    embeddings = encoder.predict(images)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    return embeddings


def find_similar_images(embedding, n=10):
    # Replace the paths to the actual paths on your device
    dataset_embeddings = np.load('C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/AE.npy')
    client = storage.Client()

    bucket = client.bucket('ezendo_images')
    embeddings_folder = 'embeddings/'

    blobs = list(bucket.list_blobs(prefix=embeddings_folder))

    image_filenames = [blob.name[len(embeddings_folder):] for blob in blobs]
    distances = np.linalg.norm(dataset_embeddings - embedding, axis=1)
    indices = np.argsort(distances)[:n]
    similar_image_urls = [f"https://storage.googleapis.com/ezendo_images/{embeddings_folder}{image_filenames[idx]}" for idx in indices]
    return similar_image_urls
