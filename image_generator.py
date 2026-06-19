import os
import torch
import numpy as np
import cv2
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline

# Speed up matrix multiplication on compatible GPUs
torch.backends.cuda.matmul.allow_tf32 = True

# --- 1. CONFIGURATION & MODELS ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# Using float16 on GPU is critical for a 2x speedup and 50% VRAM reduction
dtype = torch.float16 if device == "cuda" else torch.float32

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading Models on {device}...")
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=dtype
).to(device)

pipe_text2img = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID, torch_dtype=dtype, safety_checker=None
).to(device)

pipe_sketch2img = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL_ID, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
).to(device)

# VRAM & SPEED OPTIMIZATION
if device == "cuda":
    pipe_text2img.enable_model_cpu_offload()
    pipe_sketch2img.enable_model_cpu_offload()
else:
    pipe_text2img.enable_attention_slicing()
    pipe_sketch2img.enable_attention_slicing()

print("Models Loaded Successfully!")

# --- 2. IMAGE SLIDER DATA & LOGIC ---
SLIDER_IMAGES = [
    "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=500",  # Dog 1
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=500",  # Cat 1
    "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?w=500",  # Cat 2
    "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=500",  # Dog 2
]

def rotate_slider(current_index):
    next_index = (current_index + 1) % len(SLIDER_IMAGES)
    return SLIDER_IMAGES[next_index], next_index

# --- 3. IMAGE PROCESSING ---
def preprocess_sketch(pil_image):
    if pil_image is None:
        return None
    img = pil_image.convert('RGB')
    gray_img = img.convert('L')
    np_img = np.array(gray_img)

    edges = cv2.Canny(np_img, 100, 200)
    final_np_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_np_img).resize((512, 512))

# --- 4. PREDICTION LOGIC ---
def generate(mode, sketch_img, prompt, num_images, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Please enter a style/content prompt!")

    batch_size = int(num_images)
    processed_sketch = None

    # Duplicate the prompt to match our parallel batch size
    prompts = [prompt] * batch_size

    progress(0, desc=f"Initializing parallel generation for {batch_size} image(s)...")

    if mode == "Sketch to Image":
        if sketch_img is None:
            raise gr.Error("Please upload or draw a sketch first!")
        processed_sketch = preprocess_sketch(sketch_img)

        # Parallel execution across the GPU via batch prompts
        output_images = pipe_sketch2img(
            prompt=prompts,
            image=[processed_sketch] * batch_size,
            controlnet_conditioning_scale=1.0,
            guidance_scale=7.5,
            num_inference_steps=20  # Reduced from 30 to 20 for a 33% speedup
        ).images
    else:
        # Parallel execution across the GPU via batch prompts
        output_images = pipe_text2img(
            prompt=prompts,
            guidance_scale=7.5,
            num_inference_steps=20  # Reduced from 30 to 20 for a 33% speedup
        ).images

    # Save images locally
    for idx, image in enumerate(output_images):
        save_path = os.path.join(OUTPUT_DIR, f"generation_{idx+1}.png")
        image.save(save_path)

    preview = processed_sketch if mode == "Sketch to Image" else gr.update(visible=False)
    return preview, output_images

# --- 5. DYNAMIC UI GRADIO APP ---
with gr.Blocks(title="AI Image Studio") as demo:
    gr.Markdown("# 🎨 AI Image Generation Studio (Batch Mode)")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🐶 Live Inspo Stream 🐱")
            img_index = gr.State(value=0)
            slider_display = gr.Image(value=SLIDER_IMAGES[0], label="Slideshow", interactive=False, height=250)
            slider_timer = gr.Timer(value=3.0, active=True)

            slider_timer.tick(
                fn=rotate_slider,
                inputs=img_index,
                outputs=[slider_display, img_index],
                concurrency_limit=None
            )

    with gr.Row():
        # FIXED: Changed from 'with col := gr.Column():' to standard valid syntax
        with gr.Column() as col:
            mode = gr.Radio(
                choices=["Sketch to Image", "Text to Image"],
                value="Text to Image",
                label="1. Select Mode"
            )

            prompt = gr.Textbox(
                value="A highly detailed charcoal drawing of a futuristic engine, steam and smoke",
                label="Prompt",
                lines=3
            )
            num_images = gr.Slider(
                minimum=1, maximum=100, value=1, step=1,
                label="Number of Images to Generate (Parallel Batch)"
            )

            with gr.Group(visible=False) as sketch_inputs:
                sketch_img = gr.Image(type="pil", label="Upload or Draw Sketch", sources=["upload", "clipboard"])

            generate_btn = gr.Button("Generate Batch", variant="primary")

        with gr.Column():
            processed_preview = gr.Image(label="Processed Edge Map Preview", type="pil", visible=False)
            output_gallery = gr.Gallery(label="Generated Output Images", columns=2, rows=None, object_fit="contain")

    def update_ui(mode_selection):
        if mode_selection == "Text to Image":
            return gr.update(visible=False), gr.update(visible=False)
        return gr.update(visible=True), gr.update(visible=True)

    mode.change(fn=update_ui, inputs=mode, outputs=[sketch_inputs, processed_preview])

    generate_btn.click(
        fn=generate,
        inputs=[mode, sketch_img, prompt, num_images],
        outputs=[processed_preview, output_gallery]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)
