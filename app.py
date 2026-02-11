import gradio as gr
from utils import predict_and_decode

interface = gr.Interface(
    fn=predict_and_decode,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Captcha Decoder"
)

interface.launch()