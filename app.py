import gradio as gr
from utils import predict_and_decode
import os

example_path = "examples"
example_list = [os.path.join(example_path, f) for f in os.listdir(example_path)]

interface = gr.Interface(
    fn=predict_and_decode,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Captcha Decoder",
    examples=example_list
)

interface.launch()