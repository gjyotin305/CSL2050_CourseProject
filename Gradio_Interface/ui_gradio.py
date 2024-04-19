import gradio as gr
from PIL import Image
import os

# Define your image retrieval functions for different models
def model_1(image):

    return retrived_images

def model_2(image):
   
    return retrived_images

def model_3(image):
    
    return retrived_images

model_1_page = gr.Interface(fn=model_1,inputs=gr.Image(label="Query Image"),outputs=[gr.Image(type="pil", label="Retrieved Images") for _ in range(3)])
model_2_page = gr.Interface(fn=model_2,inputs=gr.Image(label="Query Image"),outputs=[gr.Image(type="pil", label="Retrieved Images") for _ in range(3)])
model_3_page = gr.Interface(fn=model_3,inputs=gr.Image(label="Query Image"),outputs=[gr.Image(type="pil", label="Retrieved Images") for _ in range(3)])

demo = gr.TabbedInterface([model_1_page, model_2_page,model_3_page], ["Model 1", "Model 2","Model 3"],title = "Image Retrieval System")



if __name__ == "__main__":
    demo.launch(server_name = "172.31.44.250")
