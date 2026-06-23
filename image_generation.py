import os
import csv
import torch
import numpy as np
import cv2
import uuid
import itertools
import math
import threading
import gc
from datetime import datetime
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline

# Speed up matrix multiplication on compatible GPUs
torch.backends.cuda.matmul.allow_tf32 = True

# --- 1. CONFIGURATION, LOGGING & MODELS ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

OUTPUT_DIR = "fantasy_variants"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_LOG_FILE = "generation_logs.csv"

gpu_lock = threading.Lock()
log_lock = threading.Lock()

if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Mode", "Base Prompt", "Saved Panorama/Image", "Individual Variants"])

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

if device == "cuda":
    pipe_text2img.enable_sequential_cpu_offload()
    pipe_sketch2img.enable_sequential_cpu_offload()
else:
    pipe_text2img.enable_attention_slicing()
    pipe_sketch2img.enable_attention_slicing()

print("Models Loaded Successfully!")

# --- 2. PROCEDURAL FANTASY STYLES CONFIGURATION (Creates up to 100 Combinations) ---
ATMOSPHERES = [
    "cinematic lighting", "shimmering aurora borealis sky", "dramatic thunderstorm", 
    "golden hour sunset", "mystical star-filled night sky", "steampunk industrial brass sunset", 
    "dense sun-dappled rainforest canopy", "frozen winter blizzard lighting", 
    "harsh midday sun desert heat haze", "cyberpunk deep purple and cyan neon haze"
]

ENVIRONMENTS = [
    "floating on a cloud island, waterfalls cascading into the sky", "iridescent crystal structures, ethereal magical glow",
    "jagged crackling lightning, dark epic atmosphere", "warm god rays, flying dust particles, majestic mood",
    "giant full moon, glowing liquid neon waterfalls", "complex brass gears, winding copper pipes, churning steam clouds",
    "ancient overgrown ruins, glowing emerald moss, dense jungle vines", "floating glacial mountain peaks, clear icicles",
    "surreal desert oasis, dry sand waterfalls, blazing horizon", "glowing holographic runes, futuristic dark fantasy"
]

ALL_STYLES = [f"{atm}, {env}, highly detailed, 8k resolution, digital art masterpiece" 
              for atm, env in itertools.product(ATMOSPHERES, ENVIRONMENTS)]

NEGATIVE_PROMPT = "ugly, deformed, blurry, modern buildings, cars, low quality, text, watermark, bad anatomy"

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

# --- 4. CORE PREDICTION PIPELINE ---
def generate(mode, count_selection, sketch_img, base_prompt, progress=gr.Progress()):
    warning_msg = "### ⚠️ **Notice:** AI can make mistakes. Please check important details."

    if not base_prompt.strip():
        raise gr.Error("Please enter a style or content prompt!")

    processed_sketch = None
    session_id = uuid.uuid4().hex[:8]
    
    safe_base = "".join(c for c in base_prompt if c.isalnum() or c in (' ', '_', '-')).rstrip()
    safe_base = safe_base.replace(" ", "_").lower()[:15]

    if mode == "Sketch to Image":
        if sketch_img is None:
            raise gr.Error("Please upload or draw a sketch first!")
        processed_sketch = preprocess_sketch(sketch_img)

    # --- MODE 1 & 2: SINGLE IMAGE GENERATION ---
    if mode in ["Text to Image", "Sketch to Image"]:
        progress(0.1, desc="Running single image generation pipeline...")
        with gpu_lock:
            if mode == "Sketch to Image":
                image = pipe_sketch2img(
                    prompt=base_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=processed_sketch,
                    controlnet_conditioning_scale=1.0,
                    guidance_scale=7.5,
                    num_inference_steps=20
                ).images[0]
            else:
                image = pipe_text2img(
                    prompt=base_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    guidance_scale=7.5,
                    num_inference_steps=20
                ).images[0]

        filename = f"{safe_base}_{session_id}_single.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        image.save(save_path)

        with log_lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(CSV_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, mode, base_prompt, filename, "N/A"])

        preview = processed_sketch if mode == "Sketch to Image" else gr.Image(visible=False)
        return preview, [image], warning_msg

    # --- MODE 3: FANTASY IMAGES CONTROLLABLE BATCH MATRIX LOOP ---
    else:
        generated_images = []
        saved_filenames = []
        
        target_count = int(count_selection)
        active_styles = ALL_STYLES[:target_count]
        
        for idx, style in enumerate(active_styles, start=1):
            full_prompt = f"{base_prompt}, {style}"
            progress((idx - 1) / target_count, desc=f"Generating Variant {idx}/{target_count}...")

            with gpu_lock:
                if device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()

                image = pipe_text2img(
                    prompt=full_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    guidance_scale=7.5,
                    num_inference_steps=20
                ).images[0]

            variant_filename = f"{safe_base}_{session_id}_variant_{idx}.png"
            variant_path = os.path.join(OUTPUT_DIR, variant_filename)
            image.save(variant_path)
            
            generated_images.append(image)
            saved_filenames.append(variant_filename)

        # Stitch responsive grid sheet layout dynamically based on chosen batch size
        progress(0.98, desc=f"Stitching master {target_count}-variant compilation...")
        img_w, img_h = generated_images[0].size
        
        if target_count <= 4:
            cols = target_count
        elif target_count <= 12:
            cols = 4
        else:
            cols = 10
            
        rows = math.ceil(target_count / cols)
        master_grid = Image.new('RGB', (img_w * cols, img_h * rows))

        for idx, img in enumerate(generated_images):
            col_pos = idx % cols
            row_pos = idx // cols
            master_grid.paste(img, (col_pos * img_w, row_pos * img_h))

        panorama_filename = f"{safe_base}_{session_id}_master_matrix.png"
        panorama_path = os.path.join(OUTPUT_DIR, panorama_filename)
        master_grid.save(panorama_path)

        with log_lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            variants_str = "; ".join(saved_filenames)
            with open(CSV_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, mode, base_prompt, panorama_filename, variants_str])

        output_gallery_list = [master_grid] + generated_images
        return gr.Image(visible=False), output_gallery_list, warning_msg

# --- 5. GRADIO INTERFACE CONFIGURATION ---
rotation_css = """
.gradio-container .selected-image img {
    transform: rotate(90deg) !important;
    transition: transform 0.3s ease-in-out;
    max-height: 70vh !important;
    object-fit: contain !important;
}
"""

with gr.Blocks(title="AI Multimode Image Studio", css=rotation_css) as demo:
    gr.Markdown("# 🎨 AI Multimode Image Studio")

    with gr.Row():
        with gr.Column(scale=1):
            processed_preview = gr.Image(label="Processed Edge Map Preview (Sketch mode only)", type="pil", visible=False)
            output_gallery = gr.Gallery(label="Generated Output Images", columns=4, rows=None, object_fit="contain", height=600)

    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=["Text to Image", "Sketch to Image", "Fantasy Images"],
                value="Fantasy Images",
                label="1. Choose Your Generation Mode"
            )
            
            # Interactive Slider component - now default value set to 1
            count_slider = gr.Slider(
                minimum=1, 
                maximum=100, 
                value=1, 
                step=1, 
                label="2. Slider Selector: Number of Style Variations to Generate (Only applies to 'Fantasy Images' mode)",
                interactive=True
            )

            prompt = gr.Textbox(
                value="A majestic castle sitting on top of a mountain cliffside",
                label="Core Idea (Unique environment variations will match this)",
                lines=3
            )

            with gr.Group(visible=False) as sketch_inputs:
                sketch_img = gr.Image(type="pil", label="Upload or Draw Sketch", sources=["upload", "clipboard"])

            generate_btn = gr.Button("Execute Process Pipeline", variant="primary")
            status_message = gr.Markdown("", elem_id="status-msg")

    def update_ui(mode_selection):
        if mode_selection == "Sketch to Image":
            return (
                gr.Group(visible=True),
                gr.Image(visible=True),
                gr.Slider(visible=False),
                gr.Textbox(label="Prompt (Guide your sketch details)")
            )
        elif mode_selection == "Fantasy Images":
            return (
                gr.Group(visible=False),
                gr.Image(visible=False),
                gr.Slider(visible=True),
                gr.Textbox(label="Core Idea (Unique environmental variations will match this)")
            )
        else:
            return (
                gr.Group(visible=False),
                gr.Image(visible=False),
                gr.Slider(visible=False),
                gr.Textbox(label="Prompt")
            )

    mode.change(
        fn=update_ui, 
        inputs=mode, 
        outputs=[sketch_inputs, processed_preview, count_slider, prompt]
    )

    generate_btn.click(
        fn=generate,
        inputs=[mode, count_slider, sketch_img, prompt],
        outputs=[processed_preview, output_gallery, status_message]
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(share=True)
