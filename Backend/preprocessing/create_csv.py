import os
import csv

def create_csv_from_folders(normal_folder, abnormal_folder, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_path', 'label'])

        for filename in os.listdir(normal_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(normal_folder, filename).replace('\\', '/')
                writer.writerow([img_path, 0])

        for filename in os.listdir(abnormal_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(abnormal_folder, filename).replace('\\', '/')
                writer.writerow([img_path, 1])

# replace the file paths with the actual file paths 
normal_folder_path = '/Users/Windows/Desktop/test_normal'
abnormal_folder_path = '/Users/Windows/Desktop/test_abnormal'
csv_file_path = '/Users/Windows/Desktop/labelled_dataset.csv'

create_csv_from_folders(normal_folder_path, abnormal_folder_path, csv_file_path)
