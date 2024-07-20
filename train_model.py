import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style, init
import time

# Initialize colorama
init()

# Function to convert an image to grayscale if it isn't already
def ensure_grayscale(image):
    if len(image.shape) == 3:  # If the image has 3 channels, it's not grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

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
                print(f"{Fore.CYAN}Processing file: {img_path}{Style.RESET_ALL}")
                img = cv2.imread(img_path)
                if img is None:
                    print(f"{Fore.RED}Failed to load image: {img_path}{Style.RESET_ALL}")
                else:
                    img = ensure_grayscale(img)
                    img = cv2.resize(img, (64, 128))  # Resize to a standard size
                    print(f"{Fore.CYAN}Image shape: {img.shape}{Style.RESET_ALL}")  # Debugging line
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

images, labels = load_images('dataset')

# Check if images are loaded correctly
print(f"{Fore.GREEN}Loaded {len(images)} images with labels {len(labels)}{Style.RESET_ALL}")

# Extract HOG features
def extract_hog_features(images):
    hog_features = []
    for img in images:
        print(f"{Fore.CYAN}Processing HOG for image with shape: {img.shape}{Style.RESET_ALL}")  # Debugging line
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, channel_axis=None)
        hog_features.append(fd)
        # Simple text-based animation
        print(f"{Fore.CYAN}Processing HOG feature...{Style.RESET_ALL}", end='\r')
        time.sleep(0.1)
    return np.array(hog_features)

features = extract_hog_features(images)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Train a classifier
clf = LinearSVC()
print(f"{Fore.CYAN}Training the classifier...{Style.RESET_ALL}")
clf.fit(X_train, y_train)
print(f"{Fore.GREEN}Training complete!{Style.RESET_ALL}")

# Evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"{Fore.GREEN}Accuracy: {accuracy * 100:.2f}%{Style.RESET_ALL}")
print(f"{Fore.YELLOW}Precision: {precision:.2f}{Style.RESET_ALL}")
print(f"{Fore.YELLOW}Recall: {recall:.2f}{Style.RESET_ALL}")
print(f"{Fore.YELLOW}F1 Score: {f1:.2f}{Style.RESET_ALL}")

# Save the trained model
joblib.dump(clf, 'hog_svm_model.pkl')
print(f"{Fore.GREEN}Model saved as 'hog_svm_model.pkl'{Style.RESET_ALL}")

# Plot and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
