import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from keras.models import load_model, Model

def preprocess(image_path):
    img = Image.open(image_path).resize((256, 256)).convert('L')  # Convert to grayscale
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
    return img

def load_images(csv_file):
    data = pd.read_csv(csv_file)
    images = []
    labels = data['label'].tolist()
    image_paths = data['image_path'].tolist()

    for img_path in image_paths:
        img = preprocess(img_path)
        images.append(img)

    return np.array(images), np.array(labels), image_paths

def classify_test_set(model, test_images):
    probs = model.predict(test_images)  # Predict probabilities
    predicted_classes = np.argmax(probs, axis=1)  # Determine the class with the highest probability
    confidence_scores = np.max(probs, axis=1)  # Maximum probability as confidence score
    return predicted_classes, confidence_scores

def embed_images(autoencoder, images):
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)
    encoded_images = encoder.predict(images)
    return encoded_images

def retrieve_similar_images(test_embedding, embeddings, n=10):
    distances = np.linalg.norm(embeddings - test_embedding, axis=1)
    indices = np.argsort(distances)[:n]
    return indices

def display_similar_images(indices, test_image, image_paths):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(indices) + 1, 1)
    plt.imshow(test_image.squeeze(), cmap='gray')
    plt.title("Test Image")
    plt.axis('off')

    for i, idx in enumerate(indices):
        img = Image.open(image_paths[idx])
        img = img.resize((256, 256))
        img = np.array(img) / 255.0
        plt.subplot(1, len(indices) + 1, i + 2)
        plt.imshow(img)
        plt.title(f"Similar {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_csv_file = 'C:/Users/Windows/Desktop/labelled_dataset.csv'
    autoencoder_model = 'C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/test_autoencoder_basic.keras'
    classifier_model = 'C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/original_classifier.keras'

    images, labels, image_paths = load_images(dataset_csv_file)

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1)

    autoencoder = load_model(autoencoder_model)
    classifier = load_model(classifier_model)

    # Ensure that x_test is prepared correctly as it was during training
    test_embeddings = embed_images(autoencoder, x_test)
    
    predicted_classes, confidence_scores = classify_test_set(classifier, test_embeddings)  # Note: now using the embeddings

    label_to_string = {0: 'Normal', 1: 'Abnormal'}
    for i, pred in enumerate(predicted_classes):
        print(f"Image {i}: Prediction - {label_to_string[pred]}, Confidence score - {confidence_scores[i]:.2f}")

    accuracy = accuracy_score(y_test, predicted_classes)
    precision = precision_score(y_test, predicted_classes, average='weighted')
    recall = recall_score(y_test, predicted_classes, average='weighted')
    f1 = f1_score(y_test, predicted_classes, average='weighted')

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, predicted_classes, target_names=['Normal', 'Abnormal']))

    test_image_index = 0
    test_embedding = test_embeddings[test_image_index]
    idx_closest_images = retrieve_similar_images(test_embedding, test_embeddings)

    test_image = x_test[test_image_index]
    display_similar_images(idx_closest_images, test_image, image_paths)