import cv2
import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.util import random_noise


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

def augment_images(images, labels=None):
    augmented_images = []
    augmented_labels = []

    for i, img in enumerate(images):
        label = labels[i] if labels is not None else None

        augmented_images.append(img)
        if label is not None: augmented_labels.append(label)

        horizontal_img = np.fliplr(img)
        augmented_images.append(horizontal_img)
        if label is not None: augmented_labels.append(label)

        rotated_90_img = np.rot90(img)
        augmented_images.append(rotated_90_img)
        if label is not None: augmented_labels.append(label)

        noisy_img = random_noise(img, mode='gaussian', var=0.01**2)
        augmented_images.append(noisy_img)
        if label is not None: augmented_labels.append(label)

    return (np.array(augmented_images), np.array(augmented_labels)) if labels is not None else np.array(augmented_images)

def train_model(input_shape, x_train, x_train_noisy, epochs=100, batch_size=128):
    # x_train_aug = augment_images(x_train)
    # x_train_noisy_aug = augment_images(x_train_noisy)
    input_img = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='encoded')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
     
    opt = keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=opt, loss='binary_crossentropy')
    callback = keras.callbacks.EarlyStopping(monitor='loss',  patience=50)
    autoencoder.summary()

    checkpoint = ModelCheckpoint(filepath='C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/autoencoder_basic.keras', save_best_only=True, verbose=1)
    
    autoencoder.fit(x_train_noisy, x_train, epochs=epochs, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    validation_split=0.2, 
                    callbacks=[TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False), checkpoint, callback])
    
    return autoencoder

# def train_classifier(encoder, x_train, y_train_new, epochs=200, batch_size=32):
#     x_train_encoder = encoder.predict(x_train)

#     x_train_aug, y_train_aug = augment_images(x_train_encoder, y_train_new)
#     classifier = Sequential([
#         GlobalAveragePooling2D(input_shape = x_train_aug.shape[1:]),
#         Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.25),
#         Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.25),
#         Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.25),
#         Dense(3, activation='softmax')
#     ])

#     opt = keras.optimizers.Adam(learning_rate=0.001)
#     classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#     classifier.summary()

#     class_weights = compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y_train),
#         y=y_train)
#     class_weight_dict = dict(enumerate(class_weights))

#     checkpoint = ModelCheckpoint(filepath='C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/classifier.keras', save_best_only=True, verbose=1)

#     classifier.fit(x_train_aug, y_train_aug, batch_size=batch_size, 
#                    epochs=epochs,
#                    shuffle=True,
#                    validation_split=0.2,
#                    callbacks=[TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False), checkpoint], class_weight=class_weight_dict)


# def train_model_increased_filters(input_shape, x_train, x_train_noisy, epochs=50, batch_size=64):
#     input_img = Input(shape=input_shape)
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
#     x = MaxPooling2D((2, 2), padding='same')(x)
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = MaxPooling2D((2, 2), padding='same')(x)
#     x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     encoded = MaxPooling2D((2, 2), padding='same')(x)

#     x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
#     x = UpSampling2D((2, 2))(x)
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = UpSampling2D((2, 2))(x)
#     x = Conv2D(32, (3, 3), activation='relu')(x)
#     x = UpSampling2D((2, 2))(x)
#     decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

#     autoencoder = Model(input_img, decoded)
    
#     opt = keras.optimizers.Adam(learning_rate=0.01)
#     autoencoder.compile(optimizer=opt, loss='binary_crossentropy')
#     callback = keras.callbacks.EarlyStopping(monitor='loss',  patience=20)
#     autoencoder.summary()

#     checkpoint = ModelCheckpoint(filepath="C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/autoencoder_basic.h5", save_best_only=True, verbose=1)
#     autoencoder.fit(x_train_noisy, x_train, 
#                     epochs=epochs, 
#                     batch_size=batch_size, 
#                     shuffle=True, 
#                     validation_split=0.2, 
#                     callbacks=[TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False), checkpoint])
    

if __name__ == "__main__":
    csv_file = 'C:/Users/Windows/Desktop/labelled_dataset.csv'
    images, labels = load_images(csv_file)

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1)

    y_train_new = to_categorical(y_train, num_classes=3)
    y_test_new = to_categorical(y_test, num_classes=3)

    noise_factor = 0.1
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    input_shape = (256, 256, 1)

    autoencoder = train_model(input_shape, x_train, x_train_noisy, epochs=100, batch_size=128)

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)

    # y_pred = train_classifier(encoder, x_train, y_train_new, epochs=200, batch_size=32)


    