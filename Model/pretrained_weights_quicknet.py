import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import pickle

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

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

model = create_resnet18()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=15, batch_size=64, validation_data=(test_images, test_labels))


model.save_weights('pretrained_model_weights.h5')

train_embeddings = model.predict(train_images)

num_classes = 10
mean_embeddings = np.zeros((num_classes, train_embeddings.shape[1]))

for class_label in range(num_classes):
    class_indices = np.where(train_labels.flatten() == class_label)[0]
    class_embeddings = train_embeddings[class_indices]
    mean_embedding = np.mean(class_embeddings, axis=0)
    mean_embeddings[class_label] = mean_embedding


def save_mean_embeddings(mean_embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(mean_embeddings, f)


save_mean_embeddings(mean_embeddings, 'mean_embeddings.pkl')

def classify_query(query_image):
    query_embedding = model.predict(np.expand_dims(query_image, axis=0))
    distances = [euclidean(query_embedding.flatten(), mean_embedding) for mean_embedding in mean_embeddings]
    predicted_class = np.argmin(distances)
    return predicted_class

mean_embeddings
