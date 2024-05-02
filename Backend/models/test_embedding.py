from keras.models import load_model
import numpy as np
import cv2
import time
import os

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

def embedding(encoder, alldata, embeddingFile): 
        t1 = time.time()
        learned_codes = encoder.predict(alldata)
        learned_codes = learned_codes.reshape(learned_codes.shape[0],
                                              learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[3])

        embedding_dir = os.path.dirname(embeddingFile)
        if embedding_dir and not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir, exist_ok=True)

        np.save(embeddingFile, learned_codes)
        t2 = time.time()
        print('Autoencoder-Encoding done: ', t2-t1)
        return learned_codes

if __name__ == "__main__":
    from keras.models import Model
    # Replace the file path to the actual file path in your device
    dataset = 'C:/Users/Windows/Desktop/embeddings'
    images = load_images(dataset)   

    # Replace the file path to the actual file path in your device
    autoencoder = load_model("C:/Users/Windows/Desktop/UGProject/Backend/models/checkpoints/autoencoder_basic.keras")
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)
    embeddingFile = 'C:/Users/Windows/Desktop/UGProject/Backend/models/AE.npy'
    emb = embedding(encoder, images, embeddingFile)
