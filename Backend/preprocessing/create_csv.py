import os
import csv

def create_csv_from_folders(normal_folder, abnormal_folder, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_path', 'label'])

        # Write the paths and labels for normal images
        for filename in os.listdir(normal_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(normal_folder, filename).replace('\\', '/')
                writer.writerow([img_path, 0])  # Label 0 for normal

        # Write the paths and labels for abnormal images
        for filename in os.listdir(abnormal_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(abnormal_folder, filename).replace('\\', '/')
                writer.writerow([img_path, 1])  # Label 1 for abnormal

# Usage
normal_folder_path = '/Users/Windows/Desktop/test_normal'
abnormal_folder_path = '/Users/Windows/Desktop/test_abnormal'
csv_file_path = '/Users/Windows/Desktop/test_labelled_dataset.csv'

create_csv_from_folders(normal_folder_path, abnormal_folder_path, csv_file_path)