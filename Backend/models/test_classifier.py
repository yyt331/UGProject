import cv2
import numpy as np
import pandas as pd
import keras
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.regularizers import l2
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def load_images(csv_file):
    data = pd.read_csv(csv_file)
    images = []
    labels = data['label'].tolist()

    for img_path in data['image_path']:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype('float32') / 255.
        img = np.expand_dims(img, axis=-1)
        images.append(img)

    return np.array(images), np.array(labels)

def train_classifier(encoder, x_train, y_train, epochs=100, batch_size=128):
    x_train_encoded = encoder.predict(x_train)

    classifier = Sequential([
        GlobalAveragePooling2D(input_shape=x_train_encoded.shape[1:]),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.25),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.25),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.25),
        Dense(2, activation='softmax')
    ])

    classifier.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    y_train_categorical = to_categorical(y_train, num_classes=2)

    history = classifier.fit(x_train_encoded, y_train_categorical, epochs=epochs, batch_size=batch_size,
                             validation_split=0.2, shuffle=True,
                             callbacks=[ModelCheckpoint(filepath='C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/original_classifier.keras', save_best_only=True, verbose=1),
                                        TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False)],
                             class_weight=class_weight_dict)
    return history

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_file = 'C:/Users/Windows/Desktop/labelled_dataset.csv'
    images, labels = load_images(csv_file)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1)

    autoencoder = load_model('C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/test_autoencoder_basic.keras')
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)

    history = train_classifier(encoder, x_train, y_train, epochs=100, batch_size=128)
    plot_training_history(history)