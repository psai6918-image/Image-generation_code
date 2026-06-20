import os
import csv
import torch
import numpy as np
import cv2
import uuid
import threading
from datetime import datetime
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline

# Speed up matrix multiplication on compatible GPUs
torch.backends.cuda.matmul.allow_tf32 = True

# --- 1. CONFIGURATION, LOGGING & MODELS ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# Using float16 on GPU is critical for a 2x speedup and 50% VRAM reduction
dtype = torch.float16 if device == "cuda" else torch.float32

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_LOG_FILE = "generation_logs.csv"

# Thread locks to make sure multi-user processing doesn't crash GPU or corrupt CSV file files
gpu_lock = threading.Lock()
log_lock = threading.Lock()

# Initialize the CSV file with headers if it doesn't already exist
if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Mode", "Prompt", "Num Images Request", "Saved File Names"])

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

    progress(0, desc="Waiting in queue / Initializing...")

    if mode == "Sketch to Image":
        if sketch_img is None:
            raise gr.Error("Please upload or draw a sketch first!")
        processed_sketch = preprocess_sketch(sketch_img)

    # Use the lock to ensure multi-user requests do not crash the VRAM
    with gpu_lock:
        progress(0.2, desc="Processing batch on GPU...")
        if mode == "Sketch to Image":
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

    # Save images locally using unique IDs to prevent users overwriting each other
    saved_filenames = []
    for image in output_images:
        unique_filename = f"generation_{uuid.uuid4().hex}.png"
        save_path = os.path.join(OUTPUT_DIR, unique_filename)
        image.save(save_path)
        saved_filenames.append(unique_filename)

    # Thread-safe appending to the CSV log file
    with log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Format the file list cleanly as a single string field inside the CSV
        filenames_str = "; ".join(saved_filenames)
        with open(CSV_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, mode, prompt, batch_size, filenames_str])

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
            return (
                gr.update(visible=False),       # sketch_inputs
                gr.update(visible=False),       # processed_preview
                gr.update(interactive=True)     # prompt (Enabled)
            )
        else: # Sketch to Image
            return (
                gr.update(visible=True),        # sketch_inputs
                gr.update(visible=True),        # processed_preview
                gr.update(interactive=False)    # prompt (Disabled)
            )

    # Added prompt to outputs array to handle the active/disabled logic smoothly
    mode.change(
        fn=update_ui, 
        inputs=mode, 
        outputs=[sketch_inputs, processed_preview, prompt]
    )

    generate_btn.click(
        fn=generate,
        inputs=[mode, sketch_img, prompt, num_images],
        outputs=[processed_preview, output_gallery]
    )

if __name__ == "__main__":
    # default_concurrency_limit manages simultaneous client worker pipelines
    demo.queue(default_concurrency_limit=4).launch(share=True)
