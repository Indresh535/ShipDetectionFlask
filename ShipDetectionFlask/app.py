from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
model = load_model('static/models/CNN_TrainedModel.h5')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    step_size = 40  # Default step size
    image_path = None
    ship_coordinates = []

    if request.method == 'POST':
        step_size = int(request.form.get('step_size'))
        
        if 'image' not in request.files:
            return render_template('predict.html', step_size=step_size, error='No file part')
        
        file = request.files['image']
        
        if file.filename == '':
            return render_template('predict.html', step_size=step_size, error='No selected file')
        
        if file and allowed_file(file.filename):
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
            file.save(image_path)

            # Load the input image
            image = Image.open(image_path)
            image_rgb = np.array(image.convert("RGB"))

            # Define patch and step sizes
            patch_size = 80

            # Detect ships using the trained model
            ship_candidates = []
            for y in range(0, image_rgb.shape[0] - patch_size + 1, step_size):
                for x in range(0, image_rgb.shape[1] - patch_size + 1, step_size):
                    patch = image_rgb[y:y+patch_size, x:x+patch_size]
                    input_data = patch[np.newaxis, ...]
                    prediction = model.predict(input_data)
                    if prediction[0][1] > 0.90:
                        ship_candidates.append((x, y))

            # Apply Non-Maximum Suppression
            def non_max_suppression(boxes, overlap_threshold=0.5):
                if len(boxes) == 0:
                    return []

                boxes = np.array(boxes)
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = x1 + patch_size
                y2 = y1 + patch_size

                areas = (x2 - x1 + 1) * (y2 - y1 + 1)
                scores = [1.0] * len(boxes)

                sorted_indices = np.argsort(scores)
                selected_indices = []

                while len(sorted_indices) > 0:
                    last = len(sorted_indices) - 1
                    i = sorted_indices[last]
                    selected_indices.append(i)

                    xx1 = np.maximum(x1[i], x1[sorted_indices[:last]])
                    yy1 = np.maximum(y1[i], y1[sorted_indices[:last]])
                    xx2 = np.minimum(x2[i], x2[sorted_indices[:last]])
                    yy2 = np.minimum(y2[i], y2[sorted_indices[:last]])

                    width = np.maximum(0, xx2 - xx1 + 1)
                    height = np.maximum(0, yy2 - yy1 + 1)
                    overlap = (width * height) / areas[sorted_indices[:last]]

                    sorted_indices = np.delete(sorted_indices, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

                return [ship_candidates[i] for i in selected_indices]

            # Apply NMS to get final ship coordinates
            final_ship_coordinates = non_max_suppression(ship_candidates)

            # Draw bounding boxes around detected ships
            for coord in final_ship_coordinates:
                x, y = coord
                cv2.rectangle(image_rgb, (x, y), (x + patch_size, y + patch_size), (0, 255, 0), 2)

            processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.png')
            cv2.imwrite(processed_image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            
            return render_template('predict.html', step_size=step_size, image_path=image_path, ship_coordinates=final_ship_coordinates, processed_image_path=processed_image_path)

    return render_template('predict.html', step_size=step_size, image_path=image_path, ship_coordinates=ship_coordinates)

if __name__ == '__main__':
    app.run(debug=True)
