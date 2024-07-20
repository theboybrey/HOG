import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import joblib

# Load dataset (Cat vs. Dog)
def load_images(folder):
    images = []
    labels = []
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for label, class_folder in enumerate(['cats', 'dogs']):
        class_path = os.path.join(folder, class_folder)
        for filename in os.listdir(class_path):
            ext = os.path.splitext(filename)[1]
            if ext.lower() in valid_extensions:
                img_path = os.path.join(class_path, filename)
                print(f"Processing file: {img_path}")
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                else:
                    img = cv2.resize(img, (64, 128))  # Resize to a standard size
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

images, labels = load_images('dataset')

# Check if images are loaded correctly
print(f"Loaded {len(images)} images with labels {len(labels)}")

# Extract HOG features
def extract_hog_features(images):
    hog_features = []
    for img in images:
        fd, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False)
        hog_features.append(fd)
    return np.array(hog_features)

features = extract_hog_features(images)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Train a classifier
clf = LinearSVC()
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(clf, 'hog_svm_model.pkl')
