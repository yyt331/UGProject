import cv2
import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
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

def train_model(input_shape, x_train, x_train_noisy, epochs=100, batch_size=128):
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

    checkpoint = ModelCheckpoint(filepath='C:/Users/Windows/Desktop/UGProject/Backend/autoEncoder/checkpoints/test_autoencoder_basic.keras', save_best_only=True, verbose=1)
    
    history = autoencoder.fit(x_train_noisy, x_train, epochs=epochs, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    validation_split=0.2, 
                    callbacks=[TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False), checkpoint, callback])
    
    return autoencoder, history

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

    autoencoder, history = train_model(input_shape, x_train, x_train_noisy, epochs=100, batch_size=128)

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Autoencoder Model Loss')
    plt.ylabel('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.show()



    