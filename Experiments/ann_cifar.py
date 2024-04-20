import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import tensorflow
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torchvision.models as models
import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as v2
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image,ImageFilter
import torchvision.transforms as transforms
import pickle
import os
import cv2 


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.layer_1 = nn.Linear(32 * 32 * 3, 512)  # Input size is 32x32x3
        self.relu1 = nn.ReLU()
        self.layer_2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.layer_3 = nn.Linear(256, 10)  # Output size is 10 (number of classes)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # Flatten the input tensor
        x=self.layer_1(x)
        x=self.relu1(x)
        x=self.layer_2(x)
        x=self.relu2(x)
        x=self.layer_3(x)
        return x

def retrieve(test_data,model,k_value=10):
    test_data = cv2.resize(test_data,(32,32))
    test_data = torch.tensor(test_data,dtype=torch.float32).unsqueeze(dim=0)
    print(test_data.shape)
    outputs = model(test_data)
    _, predicted = torch.max(outputs.data, 1)
    
    batch1 = unpickle(r"Model\data\data_batch_1")
    batch2 = unpickle(r"Model\data\data_batch_2")
    batch3 = unpickle(r"Model\data\data_batch_3")
    batch4 = unpickle(r"Model\data\data_batch_4")
    batch5 = unpickle(r"Model\data\data_batch_5")
    # test_batch = unpickle(r"Model\data\test_batch")
    train_batch = [batch1,batch2,batch3,batch4,batch5]

    train_data = []
    for batch in train_batch:
        y_data = batch[b'labels']
        x_data = batch[b'data']
        x_data = x_data.reshape(len(x_data),3,32,32).transpose(0,2,3,1)

        for i in range(len(y_data)):
            train_data.append((x_data[i],y_data[i]))

    x_with_specific_y = [x for x, y in train_data if y-1 == predicted]
    x_with_specific_y = np.array(x_with_specific_y)
    test_data.squeeze().permute(1,2,0)
    point = np.array(test_data.squeeze())

    distance_with_label_and_index = []

    for i,x_train in enumerate(x_with_specific_y):
        train_point = np.array(x_train[1])
        distance_with_label_and_index.append((i,np.linalg.norm(point-train_point)))

    #sorting based on distance
    distance_with_label_and_index_sorted=sorted(distance_with_label_and_index,key=lambda x: x[1])
    k_nearest_points = distance_with_label_and_index_sorted[0:k_value]

    retrived_images = []
    #calculating accuracy
    for i,(index,distance) in enumerate(k_nearest_points):
        # print(np.array(train_data[index]).shape)
        retrived_images.append(np.array(x_with_specific_y[index]))
        
    return retrived_images