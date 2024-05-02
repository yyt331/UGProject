# UGProject
Endoscopy Image Analysis Web-platform

EzEndo is a web-based platform designed to empower healthcare professionals and researchers by providing a seamless way to analyze endoscopy images. Users can upload endoscopy images onto our platform to be processed and analyzed by our advanced AI algorithms to identify if their image is normal or abnormal, and provide insights into various similar endoscopy cases. With EzEndo, we aim to enhance diagnostic accuracy, speed up the analysis process, and contribute to the advancement of gastrointestinal health care.

To test the models, first download the dataset images needed for the project in the google drive link below:

https://drive.google.com/drive/folders/16qqMpRccFR1bcxi2RjrXQXqgMVOxhtY9?usp=sharing

After downloading the image, go to the Backend folder and find the preprocessing folder. There, locate the imageProcess.py file and change the file paths to the path to your normal and abnormal sample images, as stated by the comment. Then run the file to process the images.

Then replace the file paths in the create_csv.py file and run the file. This will create a csv file with the labels for your images, with the image paths. It will be used later in the model development. Do this for the test normal and test abnormal images too for testing later.

Next, start training the autoencoder. Replace the file paths as stated in the comment in the train-ae.py file. Then run it to train the model.

Train the classifier model by doing the same. replace the file paths and run the train_classifier.py code. If you want to test the old classifier code you can do so by running the classifier.py file, also replacing the file paths stated by the comments.

To test the models, replace the file paths in the test_final.py file and run it. It will display results of the predictions and show the similar images to one of the test images.

To try running it on the Django web platform, go to the ezendo folder. There is a file called ezendo_images.json. Download the file. It will be used to access the google cloud where the images in the database are stored. 

Then, go to the ezendo folder and locate the settings.py file. Replace the file path as stated in the comment for the json file.

After that, go to the myapp folder, and then the utils.py. Replace the file paths as stated to the file paths on your device. 

Run the server by typing the command python manage.py runserver. Once the localhost server is started, add /upload/ to the url to direct to the upload page.

