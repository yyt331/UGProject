import os
import cv2

def resize_image(image, output_size):
    
    resized_image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)

    return resized_image

if __name__ == "__main__":
    # replace paths with the actual path for the file
    dataset = '/Users/Windows/Desktop/dataset'
    new_dataset = '/Users/Windows/Desktop/new_dataset'

    new_size = (256, 256)

    os.makedirs(new_dataset, exist_ok=True)

    for subdir, _, files in os.walk(dataset):
        for file in files:
            image_path = os.path.join(subdir, file)
            saveFolder = os.path.join(new_dataset, f"resized_{file}")

            resize_image(image_path, new_size, saveFolder)
