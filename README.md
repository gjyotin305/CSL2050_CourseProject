<div align="center">

# CSL2050 CourseProject

[![License](https://img.shields.io/badge/License-MIT-blue)](#license)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/gjyotin305/CSL2050_CourseProject)

</div>
  
## TOPIC : Image Retrieval


## Team Members:
  - Akshat Jain (B22CS096)
  - Harshiv Shah (B22CS098)
  - Mehta Jay Kamalkumar (B22CS034)
  - Jyotin Goel (B22AI063)
  - Rhythm Baghel (B22CS042)

## Demo


### **RiyalNet**
![Alt text](riyalnet.gif "RiyalNet")


### **QuickNet**
![Alt text](quicknet.gif "QuickNet")




## Repository Structure 

```bash
├── Experiments
│   ├── ann_cifar.py
│   ├── centroid_res18.py
│   ├── cifar_10_knn.ipynb
│   ├── cifar_10_knn_resnet18_73_percent.ipynb
│   ├── cifar_10_pca_knn.ipynb
│   ├── cifar_10_res50_m.py
│   ├── cnn-resnet34-cifar10.ipynb
│   ├── Embedding_similarity.ipynb
│   ├── HOG+KNN.ipynb
│   ├── __init__.py
│   ├── Logs
│   │   ├── RESNET50_3HLL_CIFAR.out
│   │   └── RESNET50_CIFAR.out
│   ├── PCA+HOG+KNN.ipynb
│   ├── quicknet_cifar10_centroid.ipynb
│   ├── quicknet_knn.py
│   ├── Resnet50_classification.py
│   ├── train_resnet50_3hll.py
│   └── train_resnet50_iter_1.py
├── flagged
│   └── log.csv
├── images
│   ├── akshat.jpeg
│   ├── dog.jpg
│   ├── harshiv.jpg
│   ├── horse.jpg
│   ├── image2image.png
│   ├── jay.jpeg
│   ├── jyotin.jpeg
│   ├── plane.jpg
│   ├── rhythm.jpeg
│   └── truck.jpg
├── index.html
├── __init__.py
├── LICENSE
├── MidTerm_Report.pdf
├── Model
│   ├── ann.pt
│   ├── centroid_app.py
│   ├── CIFAR.pt
│   ├── data
│   │   ├── batches.meta
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── mean_embeddings.pkl
│   │   ├── test_batch
│   │   └── train_embeddings_resnet18.pkl
│   ├── pretrained_model_weights.h5
│   ├── pretrained_weights_quicknet.py
│   ├── resnet18.h5
│   ├── Resnet50_train_features.pt
│   └── test.py
├── Preprocessing
│   ├── cifar_eda.ipynb
│   ├── k_means.py
│   └── utils.py
├── __pycache__
├── quicknet.gif
├── README.md
├── requirements.txt
├── riyalnet.gif
├── styles
│   └── style.css
└── ui_gradio.py
```
# Image Retriever Installation Guide

This guide will help you set up and install the necessary dependencies for running the project.

## Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/gjyotin305/CSL2050_CourseProject.git
2. ```bash
   cd CSL2050_CourseProject
3. ```bash
   pip install -r requirements.txt
4. ```bash
   python ui_gradio.py

This command will start the Gradio interface and display the URL where you can access it. By default, it will run on `http://127.0.0.1:7860/`. 

If you want to specify a custom IP address, you can change the argument of `demo.launch()` by inserting `server_name = "YOUR_IP_ADDRESS"`. Alternatively, you can use `share=True` to generate a public link.


