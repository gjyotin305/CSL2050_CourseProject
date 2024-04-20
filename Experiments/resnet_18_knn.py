import matplotlib.pyplot as plt
import numpy as np
import pickle

def find_knn_and_plot_image(test_data,model,k_value:int =10):
    x_test = test_data
    point = np.array(x_test)
    point = np.expand_dims(point, axis=0)
    
    test_embedding = model.predict(point)
    
    with open(r'Model\data\train_embeddings_resnet18.pkl', 'rb') as f:
        data = pickle.load(f)

    # Convert data to NumPy array
    train_data = np.array(data)

    # Convert NumPy array to TensorFlow tensor
#     data_tensor = tf.convert_to_tensor(data_array, dtype=tf.float32)

    distance_with_label_and_index = []
    print(train_data)
    for i,(x_train) in enumerate(train_data):
      train_point = np.array(x_train)
      distance_with_label_and_index.append((i,np.linalg.norm(test_embedding-train_point)))

    #sorting based on distance
    distance_with_label_and_index_sorted=sorted(distance_with_label_and_index,key=lambda x: x[1])
    k_nearest_points = distance_with_label_and_index_sorted[0:k_value]

    
    #calculating accuracy
    result =[]
    for i,(index,distance) in enumerate(k_nearest_points):
        print(index)
        result.append(train_image[index])
    return result