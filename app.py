from flask import Flask, request, jsonify
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = joblib.load('hog_svm_model.pkl')

def extract_hog_feature(image):
    fd, _ = hog(
                image, orientations=9,
                pixels_per_cell=(8, 8), 
                cells_per_block=(2, 2), 
                visualize=False,
                multichannel=False
            )
    return fd

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    feature = extract_hog_feature(img).reshape(1, -1)
    prediction = model.predict(feature)
    label = 'Dog' if prediction[0] == 1 else 'Cat'
    
    return jsonify(result=label)

if __name__ == '__main__':
    app.run(debug=True)
