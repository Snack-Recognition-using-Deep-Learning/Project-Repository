from flask import Flask, render_template, request
import os
from keras.models import load_model
from keras.preprocessing import image as keras_image
import tensorflow as tf
import cv2
import numpy as np
from shutil import move
from flask import redirect, url_for
import json

# Load your model
model = load_model('Models\\alexNet_white_2400.h5')

def normalize_image(image):
    # Normalize the image between 0 and 1
    image = image.astype(float) / 255.0
    return image

@tf.function
def predict(model, img_tensor):
    # Make the prediction
    predictions = model(img_tensor)

    # Get the class with the highest score
    score_index = tf.argmax(predictions[0])
    score = predictions[0, score_index]

    return score_index, score

# Assuming that you have a predefined mapping of class indices
class_names = {'Chipa': 0, 'Chipa guazu': 1, 'Mbeju': 2, 'Pajagua': 3, 'Pastel mandio': 4, 'Sopa': 5}
class_names = list((k) for k,v in class_names.items())
img_size = 227  # replace with the image size your model expects

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads' # Folder name to store the uploaded images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

"""
if not os.path.isdir(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
"""
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    return render_template("index.html")

# Define the mapping of class names to their respective directories
class_folders = {
    'Chipa': 'static/Chipa',
    'Chipa guazu': 'static/Chipa_guazu',
    'Mbeju': 'static/Mbeju',
    'Pajagua': 'static/Pajagua',
    'Pastel mandio': 'static/Pastel_Mandio',
    'Sopa': 'static/Sopa',
}

# Ensure each class folder exists, if not create it
for folder_path in class_folders.values():
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def switch_en(predicted_class_name):
    if predicted_class_name == "Chipa":
        return "zero"
    elif predicted_class_name == "Chipa guazu":
        return "one"
    elif predicted_class_name == "Mbeju":
        return "two"
    elif predicted_class_name == "Pajagua":
        return "three"
    elif predicted_class_name == "Pastel mandio":
        return "four"
    elif predicted_class_name == "Sopa":
        return "five"
    
def switch_es(predicted_class_name):
    if predicted_class_name == "Chipa":
        return "cero"
    elif predicted_class_name == "Chipa guazú":
        return "uno"
    elif predicted_class_name == "Mbeju":
        return "dos"
    elif predicted_class_name == "Pajagua":
        return "tres"
    elif predicted_class_name == "Pastel mandio":
        return "cuatro"
    elif predicted_class_name == "Sopa":
        return "cinco"

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "image" in request.files:
            image_file = request.files["image"]
            filename = image_file.filename  # Get the original filename of the uploaded image

            # Get bounding box from request
            bounding_box = json.loads(request.form.get('bbox'))

            # Save the image to the specified folder
            temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(temp_image_path)

            # Load the image and convert it to a NumPy array
            image = cv2.imread(temp_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (227, 227))  # For LeNet and AlexNet

            # Normalize the image
            normalized_image = normalize_image(image)

            # Convert the normalized image to a tensor
            img_tensor = tf.convert_to_tensor(normalized_image)

            # Add the batch dimension
            img_tensor = tf.expand_dims(img_tensor, 0)

            # Call the prediction function
            score_index, score = predict(model, img_tensor)

            # Get the predicted class name
            predicted_class_name = class_names[score_index]

            # Get the description for the predicted class
            class_description_en = switch_en(predicted_class_name)
            class_description_es = switch_es(predicted_class_name)
            
            # Move the uploaded image to the corresponding class folder
            final_image_path = os.path.join(class_folders[predicted_class_name], filename)
            move(temp_image_path, final_image_path)

            # Save bounding box to file
            bbox_file_path = os.path.join(class_folders[predicted_class_name], filename + '.txt')
            with open(bbox_file_path, 'w') as bbox_file:
                json.dump(bounding_box, bbox_file)
                
            # Get the list of snack labels
            labels = ["Chipa", "Chipa guazu", "Mbeju", "Pajagua", "Pastel mandio", "Sopa"]

            # Get the URL of the uploaded image
            image_url = '/' + final_image_path.replace('\\', '/')

            # Render the upload template with the uploaded image URL and the predicted class
            return render_template("upload.html", image_url=image_url, predicted_class=predicted_class_name, score=score.numpy(), labels=labels, description_en=class_description_en,description_es=class_description_es, bbox=bounding_box)
        return "No image found in the request."

    return render_template("upload.html")

@app.route('/correct', methods=['POST'])
def correct():
    correct_label = request.form.get('correct_label')
    image_url = request.form.get('image_url')

    # Convert image URL back to server path
    image_path = image_url.lstrip('/')
    image_path = os.path.join(os.getcwd(), image_path.replace('/', '\\'))

    # Move the image to the correct folder
    filename = os.path.basename(image_path)
    final_image_path = os.path.join(class_folders[correct_label], filename)
    move(image_path, final_image_path)

    # Move the bounding box file to the correct folder
    bbox_file_path = image_path + '.txt'
    final_bbox_file_path = os.path.join(class_folders[correct_label], filename + '.txt')
    move(bbox_file_path, final_bbox_file_path)

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
