import torch
import gradio as gr
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.load_lora_weights("pytorch_lora_weights.safetensors")
pipe.enable_attention_slicing()
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate(prompt):
    image = pipe(prompt).images[0]
    return image

with gr.Blocks() as demo:
    gr.Markdown("# RealVisXL + LoRA Generator")
    prompt = gr.Textbox(label="Prompt")
    output = gr.Image(type="pil")

    btn = gr.Button("Generate")
    btn.click(fn=generate, inputs=prompt, outputs=output)

demo.launch()