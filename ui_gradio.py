import gradio as gr
from PIL import Image
import os

from Experiments.Resnet50_classification import retrieve as model1_retrieve

# Define your image retrieval functions for different models
def model_1(image,num_images=3):
    retrived_images = model1_retrieve(image,k=3)
    return retrived_images

def model_2(image):
    retrived_images =None
    return retrived_images

def model_3(image):
    retrived_images =None
    return retrived_images

# num_images = gr.Slider(minimum=1, maximum=10, label="Number of Retrieved Images")
num_images = 3
model_1_page = gr.Interface(fn=model_1,inputs=gr.Image(label="Query Image"),outputs=[gr.Image(type="pil", label="Retrieved Images") for _ in range(num_images)])
model_2_page = gr.Interface(fn=model_2,inputs=gr.Image(label="Query Image"),outputs=[gr.Image(type="pil", label="Retrieved Images") for _ in range(num_images)])
model_3_page = gr.Interface(fn=model_3,inputs=gr.Image(label="Query Image"),outputs=[gr.Image(type="pil", label="Retrieved Images") for _ in range(num_images)])

demo = gr.TabbedInterface([model_1_page, model_2_page,model_3_page], ["Model 1", "Model 2","Model 3"],title = "Image Retrieval System")



if __name__ == "__main__":
    demo.launch(server_name = "172.31.44.250")
