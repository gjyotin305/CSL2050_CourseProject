import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

# Load CIFAR-10 dataset
(train_images, train_labels), (_, _) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0

# Load a pre-trained model (e.g., ResNet50)
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Extract embeddings
embeddings_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Using GlobalAveragePooling2D to reduce spatial dimensions
    layers.Dense(512, activation='relu')  # You can adjust the size of the dense layer according to your need
])

# Compute embeddings
embeddings = embeddings_model.predict(train_images)

# Initialize array to store mean embeddings for each class
num_classes = 10
mean_embeddings = np.zeros((num_classes, embeddings.shape[1]))

# Calculate mean embeddings for each class
for class_label in range(num_classes):
    class_indices = np.where(train_labels == class_label)[0]
    class_embeddings = embeddings[class_indices]
    mean_embedding = np.mean(class_embeddings, axis=0)
    mean_embeddings[class_label] = mean_embedding

print("Mean embeddings for each class:")
print(mean_embeddings)
mean_embeddings.shape
