import cv2
import os

from imageCropping import cropImage
from imageResize import resize_image

def process_image(input_folder, output_folder, output_size, val_thresh=30):

    os.makedirs(output_folder, exist_ok=True)

    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                input_path = os.path.join(subdir, file)
                img = cv2.imread(input_path)

                if img is None:
                    print(f"Unable to load image {input_path}.")
                    continue

                bbox = cropImage.getLargestBBoxArea(img, val_thresh)
                cropped_img = cropImage.crop_image(img, bbox)

                resized_image = resize_image(cropped_img, output_size)
                output_path = os.path.join(output_folder, f"cropped_{file}")

                cv2.imwrite(output_path, resized_image)

if __name__ == "__main__":
    # replace the paths to the actual paths to the files
    dataset = '/Users/Windows/Desktop/dataset'
    new_dataset = '/Users/Windows/Desktop/cropped_images'
    new_size = (256, 256)
    val_thresh = 30

    process_image(dataset, new_dataset, new_size, val_thresh)
