from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random

def kmeans_image_compression(image, n_colors=8):

    # Convert image to numpy array
    image_np = np.array(image)

    # Reshape the image to a 2D array of pixels
    image_reshaped = image_np.reshape(-1, 3)
    # print(image_reshaped.shape)

    # Shuffle the pixels
    image_reshaped_sample = shuffle(image_reshaped, random_state=0)[:image_reshaped.shape[0]]

    # Apply KMeans clustering using sklearn Library
    kmeans = KMeans(n_clusters=n_colors, random_state=0,n_init=10).fit(image_reshaped_sample)
    labels = kmeans.predict(image_reshaped)
    centroids = kmeans.cluster_centers_

    # Replace each pixel with its corresponding centroid color
    compressed_image = centroids[labels]

    # Reshape the compressed image back to its original shape
    compressed_image = compressed_image.reshape(image_np.shape)

    # Convert the compressed image array to uint8 data type
    compressed_image = compressed_image.astype(np.uint8)
    return compressed_image


compressed_train_data =[]
for x,y in train_data:
  compressed_x=kmeans_image_compression(x,6)
  compressed_train_data.append((compressed_x,y))

compressed_test_data = []
for x,y in test_data:
  compressed_x=kmeans_image_compression(x,6)
  compressed_test_data.append((compressed_x,y))