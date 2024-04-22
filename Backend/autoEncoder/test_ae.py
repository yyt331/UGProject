import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def preprocess(image_path, is_classifier=False):
    if is_classifier:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=-1)
        return image

def load_images(csv_file):
    data = pd.read_csv(csv_file)
    labels = data['label'].tolist()
    image_paths = data['image_path'].tolist()

    classifier_images, autoencoder_images = [], []
    for img_path in image_paths:
        classifier_images.append(preprocess(img_path, is_classifier=True))
        autoencoder_images.append(preprocess(img_path))

    return torch.stack(classifier_images), np.stack(autoencoder_images), np.array(labels), image_paths

def load_pytorch_classifier(model_path, device):
    model = models.alexnet(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def classify_test_set(model, test_images, device):
    test_images = test_images.to(device)
    with torch.no_grad():
        outputs = model(test_images)
        probs = F.softmax(outputs, dim=1)
    confidence_scores, predicted_classes = torch.max(probs, 1)
    return predicted_classes.cpu().numpy(), confidence_scores.cpu().numpy()

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

def embed_images(autoencoder, images):
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)
    images = images.reshape(-1, 256, 256, 1)
    embeddings = encoder.predict(images)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    return embeddings

def retrieve_similar_images(test_embedding, embeddings, n=10):
    distances = np.linalg.norm(embeddings - test_embedding, axis=1)
    indices = np.argsort(distances)[:n]
    return indices

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def display_similar_images(indices, test_image, image_paths):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(indices) + 1, 1)
    plt.imshow(test_image)
    plt.title("Test Image")
    plt.axis('off')

    # Display similar images in RGB
    for i, idx in enumerate(indices):
        img_rgb = Image.open(image_paths[idx])
        plt.subplot(1, len(indices) + 1, i + 2)
        plt.imshow(img_rgb)
        plt.title(f"Similar {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_csv_file = 'C:/Users/Windows/Desktop/labelled_dataset.csv'
    autoencoder_model = 'C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/autoencoder_basic.keras'
    classifier_model = 'C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/best_classifier.pth'
    embedding_file = 'C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/AE_colon.npy'
    embeddings = np.load(embedding_file)

    classifier_images, autoencoder_images, labels, image_paths = load_images(dataset_csv_file)

    _, x_test_classifier, _, y_test = train_test_split(classifier_images, labels, test_size=0.2, random_state=1)
    _, x_test_ae, _, _ = train_test_split(autoencoder_images, labels, test_size=0.2, random_state=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_model(autoencoder_model)
    classifier = load_pytorch_classifier(classifier_model, device)

    x_test_classifier = F.interpolate(x_test_classifier, size=(224, 224))

    predicted_classes, confidence_scores = classify_test_set(classifier, x_test_classifier, device)

    label_to_string = {0: 'Normal', 1: 'Abnormal'}
    for i, pred in enumerate(predicted_classes):
        print(f"Image {i}: Prediction - {label_to_string[pred]}, Confidence score - {confidence_scores[i]:.2f}")

    accuracy = calculate_accuracy(y_test, predicted_classes)
    print(f"Accuracy: {accuracy:.2%}")

    test_embeddings = embed_images(autoencoder, x_test_ae)

    test_image_index = 0
    test_embedding = test_embeddings[test_image_index]
    idx_closest_images = retrieve_similar_images(test_embedding, embeddings)

    test_image = x_test_classifier[test_image_index].permute(1, 2, 0).numpy()
    test_image = test_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    test_image = np.clip(test_image, 0, 1)

    display_similar_images(idx_closest_images, test_image, image_paths)
