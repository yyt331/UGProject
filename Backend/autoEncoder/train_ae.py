import os
import cv2
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split

def load_images(directory):
    images = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype('float32') / 255.
            img = np.expand_dims(img, axis=-1)
            images.append(img)

    return np.array(images)

def train_model(input_shape, x_train, x_train_noisy, epochs=50, batch_size=128):
    input_img = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    autoencoder.summary()

    checkpoint = ModelCheckpoint(filepath='./checkpoints/autoencoder_basic.h5', save_best_only=True, verbose=1)
    autoencoder.fit(x_train_noisy, x_train, epochs=epochs, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    validation_split=0.2, 
                    callbacks=[TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False), checkpoint])


def train_model_increased_filters(input_shape, x_train, x_train_noisy, epochs=50, batch_size=128):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    autoencoder.summary()

    checkpoint = ModelCheckpoint(filepath='./checkpoints/autoencoder_increased_filters.h5', save_best_only=True, verbose=1)
    autoencoder.fit(x_train_noisy, x_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    validation_split=0.2, 
                    callbacks=[TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False), checkpoint])
    

if __name__ == "__main__":
    dataset = '/Users/Windows/Desktop/cropped_images'
    images = load_images(dataset)

    x_train, x_test = train_test_split(images, test_size=0.2, random_state=1)

    noise_factor = 0.25
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    input_shape = (256, 256, 1)

    train_model(input_shape, x_train, x_train_noisy, epochs=50, batch_size=128)