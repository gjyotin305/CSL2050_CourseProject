
import pickle
import random
import numpy as np
import cv2
from tensorflow.keras import models, layers

# Load the trained model architecture
def create_resnet18():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Load the pretrained weights
def load_pretrained_weights(model, weights_path):
    model.load_weights(weights_path)

# Function to unpickle a file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Function to load images from the unpickled data batch file of a specific class
def load_class_images(class_index, train_batches):
    images = []
    for batch in train_batches:
        if b'data' in batch and b'labels' in batch:
            data = batch[b'data']
            labels = batch[b'labels']
            for i, label in enumerate(labels):
                if label == class_index:
                    img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)  # Reshape and transpose the image
                    images.append(img)
    return images

# Function to calculate the distance of the mean embeddings with a query image
def classify_query(query_image, model, mean_embeddings):
    query_embedding = model.predict(np.expand_dims(query_image, axis=0))
    distances = [np.linalg.norm(query_embedding.flatten() - mean_embedding) for mean_embedding in mean_embeddings]
    predicted_class = np.argmin(distances)
    return predicted_class


def retrieve(query_image,k=3):
    model = create_resnet18()
    load_pretrained_weights(model, 'Model\pretrained_model_weights.h5')

    mean_embeddings = pickle.load(open('Model\data\mean_embeddings.pkl', 'rb'))

    # query_image_path = '/content/airplane_8925.png'
    # query_image = cv2.imread(query_image_path)
    query_image = cv2.resize(query_image, (32, 32)) / 255.0  # Resize and normalize the image

    predicted_class = classify_query(query_image, model, mean_embeddings)
    # print("Predicted Class:", predicted_class)

    # Load random images of the predicted class
    train_batches = [unpickle(rf"Model\data\data_batch_{i}") for i in range(1,6)]
    class_images = load_class_images(predicted_class, train_batches)
    
    if class_images:
        random_images = random.sample(class_images, k)  # Select 3 random images
        return random_images
    else:
        print("No images found for the predicted class.")


