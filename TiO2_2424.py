import numpy as np
import gzip
import cv2
import os

def load_mnist_labels(label_path):
    with gzip.open(label_path, 'rb') as f:
        # Skip the magic number and count
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_mnist_images(image_path):
    with gzip.open(image_path, 'rb') as f:
        f.read(16)  # Skip the magic number and dimensions
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    return images

def extract_center(images):
    # Remove 2 pixels from each side (left, right, top, bottom) to get a 24x24 center part
    center_images = images[:, 2:26, 2:26]
    return center_images

def save_center_images_with_labels(images, labels, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, img in enumerate(images):
        file_path = os.path.join(save_dir, f"label_{labels[i]}_image_{i}.png")
        cv2.imwrite(file_path, img)

# Paths
image_path_train = r'C:\Users\97843\Desktop\MNIST\t10k-images-idx3-ubyte.gz'
label_path_train = r'C:\Users\97843\Desktop\MNIST\t10k-labels-idx1-ubyte.gz'
save_image_dir = r'C:\Users\97843\Desktop\MNIST_2\10k'

# Load data
images_train = load_mnist_images(image_path_train)
labels_train = load_mnist_labels(label_path_train)

# Process images
center_images_train = extract_center(images_train)

# Save the center 24x24 images as PNG files with labels
save_center_images_with_labels(center_images_train, labels_train, save_image_dir)
