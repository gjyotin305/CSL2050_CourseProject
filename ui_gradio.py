import gradio as gr
from PIL import Image
import os
import torch

from Experiments.Resnet50_classification import retrieve as model1_retrieve
from Model.centroid_app import retrieve as model2_retrieve


def model_1(image, num_images=3):
    retrived_images = model1_retrieve(image, k=num_images)
    return retrived_images


def model_2(image,num_images):
    retrived_images = model2_retrieve(image,k=num_images)
    return retrived_images

model_1_page = gr.Interface(
    fn=model_1,
    inputs=[gr.Image(
        label="Query Image"),gr.Number(label="Number of Images")],
    outputs=gr.Gallery( type="pil",
            label="Retrieved Images"),title = "RiyalNet - This Model has the best accuracy of 97 %")

model_2_page = gr.Interface(
    fn=model_2,
    inputs=[gr.Image(
        label="Query Image"),gr.Number(label="Number of Images")],
    outputs=gr.Gallery( type="pil",
            label="Retrieved Images"),title="QuickNet - This Model has the best runtime")

demo = gr.TabbedInterface([model_1_page, model_2_page], 
                          ["RiyalNet", "QuickNet"], 
                          title="Image Retrieval System")


if __name__ == "__main__":
    demo.launch(server_name="172.31.44.250")
