import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_images(csv_file):
    data = pd.read_csv(csv_file)
    images = []
    labels = data['label'].tolist()
    image_paths = data['image_path'].tolist()

    label_mapping = {0.0: 0, 0.5: 1, 1.0: 1}
    mapped_labels = [label_mapping[label] for label in labels]

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.
        img = np.expand_dims(img, axis=-1)
        images.append(img)

    return np.array(images), np.array(mapped_labels), image_paths

def load_pytorch_classifier(model_path, device):
    model = models.alexnet(pretrained=True)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def classify_test_set(model, test_images, device):
    test_images_tensor = torch.tensor(test_images, device=device)
    outputs = model(test_images_tensor)
    probs = F.softmax(outputs, dim=1)
    confidence_scores, predicted_classes = torch.max(probs, 1)
    predicted_classes = predicted_classes.detach().cpu().numpy()
    confidence_scores = confidence_scores.detach().cpu().numpy()
    return predicted_classes, confidence_scores

# def classify_test_set(classifier, test_images, threshold=0.3):
#     test_images_tensor = torch.tensor(test_images, dtype=torch.float32)
#     with torch.no_grad():
#         predictions = classifier(test_images_tensor)
#         predicted_probs = F.softmax(predictions, dim=1)
#         confidence_score, predicted_class = torch.max(predicted_probs, dim=1)
        
#     confidence_score_np = confidence_score.numpy()
#     predicted_class_np = predicted_class.numpy()
#     predicted_class_np[confidence_score_np < threshold] = -1
    
#     return predicted_class_np, confidence_score_np

def retrieve_similar_images(test_image, encoded_images, n=10):
    test_image_reshaped = test_image.reshape(1, 32, 32, 8)
    distances = np.array([np.linalg.norm(encoded_image.flatten() - test_image_reshaped.flatten()) for encoded_image in encoded_images])
    idx_closest = distances.argsort()[:n]
    return idx_closest

def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def display_similar_images(indices, test_image, image_paths):
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 4, 1)
    plt.imshow(test_image)
    plt.title("Test Image")
    for i, idx in enumerate(indices, start=1):
        img_path = image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(3, 4, i+1)
        plt.imshow(img)
        plt.title(f"Similar {i}")
    plt.show()

if __name__ == "__main__":
    dataset_csv_file = 'C:/Users/Windows/Desktop/dataset_labelled.csv'
    autoencoder_model = 'C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/autoencoder_basic.keras'
    classifier_model = 'C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/classifier.pth'
    embedding_file = 'C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/AE_colon.npy'
    embeddings = np.load(embedding_file)

    images, labels, image_paths = load_images(dataset_csv_file)

    _, x_test, _, y_test = train_test_split(images, labels, test_size=0.4, random_state=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_model(autoencoder_model)
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)
    classifier = load_pytorch_classifier(classifier_model, device)

    if x_test.ndim == 5:  
        x_test = np.squeeze(x_test, axis=-1)

    x_test_gray = np.dot(x_test[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)  # Convert to grayscale
    x_test_gray = np.expand_dims(x_test_gray, axis=-1)

    test_images = encoder.predict(x_test_gray).reshape(-1, 32*32*8)
    test_images = test_images.astype(np.float32) 
    x_test_resized = [cv2.resize(img, (224, 224)) for img in x_test.squeeze()] 
    x_test_resized = np.stack(x_test_resized).reshape((-1, 3, 224, 224)) 
    threshold = 0.3
    predicted_class, confidence_score = classify_test_set(classifier, x_test_resized, device)

    label_to_string = {0: 'normal', 1: 'abnormal'}
    predicted_labels = [label_to_string[pred] for pred in predicted_class]

    accuracy = calculate_accuracy(y_test, predicted_class)
    print(f"Accuracy: {accuracy:.2%}")

    for i, label in enumerate(predicted_labels):
        print(f"Image {i}: Prediction - {label}, Confidence score - {confidence_score[i]:.2f}")

    chosen_test_image = test_images[1]
    idx_closest_images = retrieve_similar_images(chosen_test_image, embeddings)
    display_similar_images(idx_closest_images, x_test[1], image_paths)