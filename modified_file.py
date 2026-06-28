import os
import csv
import torch
import numpy as np
import cv2
import uuid
import itertools
import math
import gc
from datetime import datetime
from PIL import Image
import gradio as gr
from diffusers import (
    StableDiffusionPipeline, 
    ControlNetModel, 
    StableDiffusionControlNetPipeline, 
    StableDiffusionImg2ImgPipeline
)

# --- 1. CONFIGURATION, LOGGING & MODELS ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

OUTPUT_DIR = "fantasy_variants"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_LOG_FILE = "generation_logs.csv"

if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time Start", 
            "Time End", 
            "Duration (Minutes)", 
            "Mode", 
            "Base Prompt", 
            "Saved Panorama/Image", 
            "Individual Variants"
        ])

print(f"Loading Models on {device} ({dtype})...")
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Load the core shared components FIRST
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=dtype
)

# Pipe 1: Load Text to Image completely from scratch
pipe_text2img = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
)

# CRITICAL FIX: Extract components directly from Pipe 1 to build Pipe 2 and Pipe 3.
# This prevents duplicating 8+ Gigabytes of weights in your RAM/VRAM!
shared_components = pipe_text2img.components

pipe_sketch2img = StableDiffusionControlNetPipeline(
    **shared_components,
    controlnet=controlnet
)

pipe_img2img = StableDiffusionImg2ImgPipeline(
    **shared_components
)

# Push everything directly onto the GPU hardware
pipe_text2img.to(device)
pipe_sketch2img.to(device)
pipe_img2img.to(device)

if device == "cuda":
    try:
        pipe_text2img.enable_xformers_memory_efficient_attention()
        pipe_sketch2img.enable_xformers_memory_efficient_attention()
        pipe_img2img.enable_xformers_memory_efficient_attention()
        print("XFormers memory efficiency activated successfully.")
    except Exception:
        torch.backends.cuda.enable_flash_sdp(True)
        print("Flash Attention/SDPA activated natively.")

print("Models Loaded Successfully with Shared Memory Map!")

# --- 2. PROCEDURAL FANTASY STYLES CONFIGURATION ---
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
    start_dt = datetime.now()
    warning_msg = '<div style="text-align: center; width: 100%; font-size: 1.1em; color: #888; background: transparent; border: none; padding: 0; margin: 10px 0;">⚠ AI can make mistakes. Please verify important outputs.</div>'

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

    target_count = int(count_selection)
    generated_images = []
    saved_paths = []

    # DOUBLE-CHECK DEVICE VALIDITY BEFORE STARTING RENDERS
    current_pipe = pipe_sketch2img if mode == "Sketch to Image" else pipe_text2img
    if device == "cuda" and str(current_pipe.device) == "cpu":
        current_pipe.to("cuda")

    with torch.inference_mode():
        for idx in range(1, target_count + 1):
            progress((idx - 1) / target_count, desc=f"Generating Image {idx}/{target_count}...")
            
            if mode == "Fantasy Images":
                current_style = ALL_STYLES[(idx - 1) % len(ALL_STYLES)]
                full_prompt = f"{base_prompt}, {current_style}"
            else:
                full_prompt = base_prompt

            if mode == "Sketch to Image":
                image = pipe_sketch2img(
                    prompt=full_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=processed_sketch,
                    controlnet_conditioning_scale=1.0,
                    guidance_scale=7.5,
                    num_inference_steps=20
                ).images[0]
            else:
                image = pipe_text2img(
                    prompt=full_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    guidance_scale=7.5,
                    num_inference_steps=20
                ).images[0]

            variant_filename = f"{safe_base}_{session_id}_frame_{idx}.png"
            variant_path = os.path.abspath(os.path.join(OUTPUT_DIR, variant_filename))
            image.save(variant_path)
            
            generated_images.append(image)
            saved_paths.append(variant_path)

            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

    progress(0.95, desc=f"Assembling image matrix canvas...")
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
    panorama_path = os.path.abspath(os.path.join(OUTPUT_DIR, panorama_filename))
    master_grid.save(panorama_path)

    end_dt = datetime.now()
    duration_minutes = round((end_dt - start_dt).total_seconds() / 60.0, 3)

    variants_str = "; ".join(saved_paths)
    with open(CSV_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            start_dt.strftime("%Y-%m-%d %H:%M:%S"), 
            end_dt.strftime("%Y-%m-%d %H:%M:%S"), 
            duration_minutes, mode, base_prompt, panorama_filename, variants_str
        ])

    if target_count == 1:
        output_gallery_list = saved_paths
    else:
        output_gallery_list = [panorama_path] + saved_paths

    preview_out = saved_paths[0] if mode == "Sketch to Image" else None
    return gr.update(value=preview_out, visible=(preview_out is not None)), output_gallery_list, warning_msg, output_gallery_list


# --- 5. IMAGE-TO-IMAGE SELECTION & WORKSPACE MODIFICATION ---
def on_gallery_select(evt: gr.SelectData, current_images):
    if not current_images or evt.index is None:
        return gr.update(visible=False), None, None, None

    selected_idx = evt.index
    if selected_idx < len(current_images):
        selected_image_source = current_images[selected_idx]
    else:
        selected_image_source = current_images[0]

    if isinstance(selected_image_source, dict):
        selected_image_source = selected_image_source.get("name", "")

    return gr.update(visible=True), selected_image_source, "", selected_image_source


def modify_selected_image(base_image, modify_prompt, strength_slider, base_prompt_input, progress=gr.Progress()):
    if base_image is None:
        raise gr.Error("No source image selected! Click an image in the gallery first.")
    if not modify_prompt.strip():
        raise gr.Error("Please describe what changes or elements to introduce!")

    progress(0.2, desc="Executing image modification workflow...")

    if isinstance(base_image, str):
        base_image_pil = Image.open(base_image)
    else:
        base_image_pil = Image.fromarray(base_image)

    refined_prompt = f"({modify_prompt}:1.3) BREAK, based on the composition of {base_prompt_input}, highly detailed masterpiece"

    if device == "cuda" and str(pipe_img2img.device) == "cpu":
        pipe_img2img.to("cuda")

    with torch.inference_mode():
        modified_img = pipe_img2img(
            prompt=refined_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=base_image_pil.convert("RGB"),
            strength=float(strength_slider),
            guidance_scale=9.5, 
            num_inference_steps=20
        ).images[0]
    
    session_id = uuid.uuid4().hex[:5]
    filename = f"modified_{session_id}.png"
    save_path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
    modified_img.save(save_path)

    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    return save_path, '<div style="color: #16a34a; font-weight: bold;">✨ Workspace modifications applied! Check your target window.</div>'


def reset_workspace_image(backup_path):
    if not backup_path:
        return gr.update(), '<div style="color: #dc2626;">No original image available to reset to.</div>'
    return backup_path, '<div style="color: #2563eb;">🔄 Reset workspace back to your original base image. Try altering strength values!</div>'


def set_processing_notice():
    return (
        '<div style="text-align: center; width: 100%; font-weight: bold; color: #d97706; background: transparent; border: none; padding: 0; margin: 10px 0;">⏳ Processing Pipeline Initiated... <span style="font-weight: normal; font-size: 0.9em; color: #555;">(AI can make mistakes. Generating asset frames...)</span></div>',
        gr.update(visible=False)
    )


def append_to_favorites(target_img, custom_name, current_favorites):
    if not target_img:
        return current_favorites, gr.update(), '<div style="color: #dc2626;">Select or modify an image before archiving.</div>'
    
    try:
        if isinstance(target_img, str):
            base_image_pil = Image.open(target_img)
        else:
            base_image_pil = Image.fromarray(target_img)
        
        clean_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '_', '-')).strip()
        if not clean_name:
            clean_name = f"favorite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        clean_name = clean_name.replace(" ", "_") + ".png"
        favorite_path = os.path.abspath(os.path.join(OUTPUT_DIR, clean_name))
        
        base_image_pil.save(favorite_path)
        
        if favorite_path not in current_favorites:
            current_favorites.append(favorite_path)
            status = f'<div style="color: #16a34a; font-weight: bold;">💖 Saved as "{clean_name}" into Favorites Scrapbook!</div>'
        else:
            status = f'<div style="color: #d97706;">Image is already archived under this configuration pathway.</div>'
            
    except Exception as e:
        return current_favorites, gr.update(), f'<div style="color: #dc2626;">Failed saving file asset: {str(e)}</div>'
        
    return current_favorites, gr.update(value=current_favorites, visible=True), status


def filter_favorites(query, current_favorites):
    if not query.strip():
        return gr.update(value=current_favorites)
    
    q = query.lower().strip()
    filtered = [path for path in current_favorites if q in os.path.basename(path).lower()]
    return gr.update(value=filtered)


# --- 6. GRADIO INTERFACE CONFIGURATION & CUSTOM STYLE RULES ---
GENERATOR_CSS = """
.gradio-container, .gradio-row, .gradio-column {
    height: auto !important;
}
.gradio-container .selected-image img, 
.gradio-container img {
    transform: rotate(0deg) !important;
    max-height: none !important;
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}
#saved_favorites_gallery img {
    max-height: 180px !important;
    max-width: 180px !important;
    width: auto !important;
    height: auto !important;
    margin: 0 auto !important;
    object-fit: contain !important;
    border-radius: 8px !important;
}
#saved_favorites_gallery .gallery-item {
    background: transparent !important;
    box-shadow: none !important;
    padding: 4px !important;
}
.prose.gradio-html, .gradio-html div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
footer, .built-with, .prose a[href*="gradio.app"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}
"""

def update_ui(mode_selection):
    if mode_selection == "Sketch to Image":
        return (
            gr.update(visible=True), 
            gr.update(visible=True), 
            gr.update(interactive=True),
            gr.update(label="Prompt (Guide your sketch details)")
        )
    elif mode_selection == "Fantasy Images":
        return (
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(interactive=True),
            gr.update(label="Core Idea (Unique environmental variations will match this)")
        )
    else:
        return (
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(interactive=True),
            gr.update(label="Prompt (Describe the image you want to generate)")
        )

with gr.Blocks(title="AI Multimode Image Studio") as demo:
    generated_cache = gr.State([])
    original_image_backup = gr.State(None)
    favorites_cache = gr.State([]) 
    
    gr.Markdown("# 🎨 AI Multimode Image Studio")

    with gr.Tabs():
        with gr.TabItem("Studio Workspace"):
            with gr.Group() as studio_container:
                with gr.Row():
                    with gr.Column(scale=2):
                        processed_preview = gr.Image(label="Processed Edge Map Preview (Sketch mode only)", type="filepath", visible=False)
                        output_gallery = gr.Gallery(label="Generated Output Images (Click any photo below to modify it)", columns=4, rows=None, object_fit="contain", height="auto", type="filepath")
                    
                    with gr.Column(scale=1):
                        with gr.Group(visible=False) as modify_panel:
                            gr.Markdown("### 🛠️ Modify Selected Variant Workspace")
                            selected_preview = gr.Image(label="Target Workspace Image", type="filepath", interactive=False)
                            
                            with gr.Row():
                                reset_btn = gr.Button("🔄 Revert Base", size="sm", variant="secondary")
                                save_favorite_btn = gr.Button("💖 Save to Favorites", size="sm", variant="primary")
                            
                            custom_filename_input = gr.Textbox(
                                label="Custom Save Name (Optional)", 
                                placeholder="e.g., retro_futuristic_car (leaves empty for timestamp)",
                                max_lines=1
                            )
                            
                            modify_input_prompt = gr.Textbox(
                                label="What elements or changes would you like to add?", 
                                placeholder="e.g., 'deep red water, bloody river cascade'"
                            )
                            
                            strength_control = gr.Slider(
                                minimum=0.10, maximum=0.90, value=0.45, step=0.05,
                                label="Transformation Strength (Lower preserves shapes, Higher injects more new elements)"
                            )
                            submit_modification_btn = gr.Button("Apply Workspace Prompt Modifications", variant="secondary")
                            modification_status = gr.HTML("")

                with gr.Row():
                    with gr.Column(scale=1):
                        mode = gr.Radio(
                            choices=["Text to Image", "Sketch to Image", "Fantasy Images"],
                            value="Text to Image",
                            label="1. Choose Your Generation Mode"
                        )
                        
                        count_slider = gr.Slider(
                            minimum=1, 
                            maximum=100, 
                            value=1, 
                            step=1, 
                            label="2. Slider Selector: Number of Style Variations to Generate",
                            interactive=True
                        )

                        prompt = gr.Textbox(
                            value="",
                            label="Prompt (Describe the image you want to generate)",
                            lines=3
                        )

                        with gr.Group(visible=False) as sketch_inputs:
                            sketch_img = gr.Image(type="pil", label="Upload or Draw Sketch", sources=["upload", "clipboard"])

                        generate_btn = gr.Button("Execute Process Pipeline", variant="primary")
                        status_message = gr.HTML("")

        with gr.TabItem("Saved Favorites Scrapbook"):
            with gr.Group():
                gr.Markdown("### 🌟 Saved Favorites Gallery")
                gr.Markdown("Images you save inside the workspace module by pressing **💖 Save to Favorites** accumulate automatically below.")
                
                search_bar = gr.Textbox(
                    label="🔍 Filter Favorites by Name", 
                    placeholder="Type to search your saved assets instantly...",
                    max_lines=1
                )
                
                saved_gallery = gr.Gallery(
                    elem_id="saved_favorites_gallery",
                    label="Your Collected Artifact Scrapbook", 
                    columns=6, 
                    rows=None, 
                    object_fit="contain", 
                    height="auto", 
                    type="filepath",
                    visible=False
                )

        # --- CONTROLLER HOOKS & WIRING ---
        search_bar.change(
            fn=filter_favorites,
            inputs=[search_bar, favorites_cache],
            outputs=[saved_gallery]
        )

        mode.change(
            fn=update_ui, 
            inputs=mode, 
            outputs=[sketch_inputs, processed_preview, count_slider, prompt]
        )

        generate_btn.click(
            fn=set_processing_notice,
            inputs=None,
            outputs=[status_message, modify_panel]
        ).then(
            fn=generate,
            inputs=[mode, count_slider, sketch_img, prompt],
            outputs=[processed_preview, output_gallery, status_message, generated_cache]
        )

        output_gallery.select(
            fn=on_gallery_select,
            inputs=[generated_cache],
            outputs=[modify_panel, selected_preview, modification_status, original_image_backup]
        )

        reset_btn.click(
            fn=reset_workspace_image,
            inputs=[original_image_backup],
            outputs=[selected_preview, modification_status]
        )

        save_favorite_btn.click(
            fn=append_to_favorites,
            inputs=[selected_preview, custom_filename_input, favorites_cache],
            outputs=[favorites_cache, saved_gallery, modification_status]
        )

        submit_modification_btn.click(
            fn=lambda: '<div style="color: #d97706; font-weight: bold;">⏳ Rendering modifications inside workspace window...</div>',
            inputs=None,
            outputs=[modification_status]
        ).then(
            fn=modify_selected_image,
            inputs=[selected_preview, modify_input_prompt, strength_control, prompt],
            outputs=[selected_preview, modification_status]
        )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        share=True, 
        server_name="0.0.0.0")
