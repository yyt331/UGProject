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
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, classification_report
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

def load_images(csv_file, is_classifier=False):
    data = pd.read_csv(csv_file)
    labels = data['label'].tolist() if 'label' in data.columns else None
    image_paths = data['image_path'].tolist()

    images = []
    for img_path in image_paths:
        images.append(preprocess(img_path, is_classifier=is_classifier))

    images = torch.stack(images) if is_classifier else np.stack(images)
    return images, np.array(labels), image_paths

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
    return predicted_classes.cpu().numpy(), confidence_scores.cpu().numpy(), probs.cpu().numpy()

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

    for i, idx in enumerate(indices):
        img_rgb = Image.open(image_paths[idx])
        plt.subplot(1, len(indices) + 1, i + 2)
        plt.imshow(img_rgb)
        plt.title(f"Similar {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace the file paths to the actual file paths in your device
    dataset_csv = 'C:/Users/Windows/Desktop/labelled_dataset.csv'
    test_dataset_csv = 'C:/Users/Windows/Desktop/test_labelled_dataset.csv'
    autoencoder_model = 'C:/Users/Windows/Desktop/UGProject/Backend/models/checkpoints/autoencoder_basic.keras'
    classifier_model = 'C:/Users/Windows/Desktop/UGProject/Backend/models/checkpoints/best_classifier.pth'
    embedding_file = 'C:/Users/Windows/Desktop/UGProject/Backend/models/AE.npy'
    embeddings = np.load(embedding_file)

    x_test_classifier, y_test, test_image_paths = load_images(test_dataset_csv, is_classifier=True)
    _, _, image_paths = load_images(dataset_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_model(autoencoder_model)
    classifier = load_pytorch_classifier(classifier_model, device)

    x_test_classifier = F.interpolate(x_test_classifier, size=(224, 224))

    predicted_classes, confidence_scores, predicted_probs = classify_test_set(classifier, x_test_classifier, device)

    probabilities = predicted_probs[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

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

    x_test_ae, _, _ = load_images(test_dataset_csv)
    test_embeddings = embed_images(autoencoder, x_test_ae)

    test_image_index = 0
    test_embedding = test_embeddings[test_image_index]
    idx_closest_images = retrieve_similar_images(test_embedding, embeddings)

    test_image = x_test_classifier[test_image_index].permute(1, 2, 0).numpy()
    test_image = test_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    test_image = np.clip(test_image, 0, 1)

    display_similar_images(idx_closest_images, test_image, image_paths)
