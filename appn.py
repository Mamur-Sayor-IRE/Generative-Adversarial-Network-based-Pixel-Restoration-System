import os
from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load your TensorFlow model in .h5 format
model_path = r'E:\leaf\model2.h5'  # Update the path to your .h5 model
model = tf.keras.models.load_model(model_path)

# Replace this with your actual list of class names
classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Function for model inference
def predict_image(image_path):
    try:
        # Read and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)
        print("Raw Prediction:", prediction)

        # Get the predicted class index and name
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = classes[predicted_class_index]

        return f"Predicted Class: {predicted_class_name}"
    except Exception as e:
        return f"Error predicting image: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # If the user submits an empty file
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # If the file is valid
        if file:
            # Save the uploaded file
            file_path = 'static/uploaded_image.jpg'  # Save the image in the "static" folder
            file.save(file_path)

            # Perform prediction and get the result
            prediction_result = predict_image(file_path)

            # Render the result template
            return render_template('result.html', image_path=file_path, prediction=prediction_result)

    return render_template('index.html', message='Upload your image for prediction')

if __name__ == '__main__':
    app.run(debug=True)
