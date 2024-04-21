# CSL2050 CourseProject

## TOPIC : Image Retrieval

## Team Members:
  - Akshat Jain (B22CS096)
  - Harshiv Shah (B22CS098)
  - Mehta Jay Kamalkumar (B22CS034)
  - Jyotin Goel (B22AI063)
  - Rhythm Baghel (B22CS042)

## Repository Structure 

```bash
├── Experiments
│   ├── cifar_10_knn.ipynb
│   ├── cifar_10_pca_knn.ipynb
│   ├── cnn-resnet34-cifar10.ipynb
│   ├── cifar_10_mean.ipynb
│   ├── cifar_10_centroid.ipynb
│   ├── HOG+KNN.ipynb
│   ├── Logs
│   │   └── RESNET50_CIFAR.out
│   ├── PCA+HOG+KNN.ipynb
│   └── train.py
├── LICENSE
├── MidTerm_Report.pdf
├── Model
│   ├── CIFAR.pt
│   └── test.py
├──Gradio_Interface
│   ├── ui_gradio.py
├── Preprocessing
│   ├── cifar_eda.ipynb
│   ├── k_means.py
│   └── utils.py
└── README.md
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
   python .\ui_gradio.py

This command will start the Gradio interface and display the URL where you can access it. By default, it will run on `http://127.0.0.1:7860/`. 

If you want to specify a custom IP address, you can change the argument of `demo.launch()` by inserting `server_name = "YOUR_IP_ADDRESS"`. Alternatively, you can use `share=True` to generate a public link.
