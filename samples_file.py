import os
import csv
import torch
import numpy as np
import cv2
import uuid
import itertools
import math
import gc
import re
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
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.cuda.empty_cache()

OUTPUT_DIR = "fantasy_variants"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_LOG_FILE = "generation_logs.csv"

if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time Start", "Time End", "Duration (Minutes)",
            "Mode", "Base Prompt", "Saved Panorama/Image", "Individual Variants"
        ])

print(f"Loading Models on {device} ({dtype})...")
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=dtype
)

pipe_text2img = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
)

shared_components = pipe_text2img.components

pipe_sketch2img = StableDiffusionControlNetPipeline(
    **shared_components,
    controlnet=controlnet
)

pipe_img2img = StableDiffusionImg2ImgPipeline(
    **shared_components
)

pipe_text2img.to(device)
pipe_sketch2img.to(device)
pipe_img2img.to(device)

if device == "cuda":
    pipe_text2img.enable_attention_slicing()
    pipe_sketch2img.enable_attention_slicing()
    pipe_img2img.enable_attention_slicing()

# --- 2. PROCEDURAL FANTASY STYLES ---
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

def preprocess_sketch(pil_image):
    if pil_image is None:
        return None
    img = pil_image.convert('RGB')
    gray_img = img.convert('L')
    np_img = np.array(gray_img)
    edges = cv2.Canny(np_img, 100, 200)
    final_np_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_np_img).resize((512, 512))

# --- 3. PIPELINE EXECUTIONS ---
def generate(mode, count_selection, sketch_img, base_prompt, progress=gr.Progress()):
    start_dt = datetime.now()
    warning_msg = '<div style="text-align: center; width: 100%; font-size: 1.1em; color: #fff; background: transparent; padding: 0; margin: 10px 0;">⚠ Verify important outputs before saving.</div>'

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

    target_count = max(1, int(count_selection))
    saved_paths = []

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
            saved_paths.append(variant_path)

    end_dt = datetime.now()
    duration_minutes = round((end_dt - start_dt).total_seconds() / 60.0, 3)

    variants_str = "; ".join(saved_paths)
    with open(CSV_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            start_dt.strftime("%Y-%m-%d %H:%M:%S"), end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            duration_minutes, mode, base_prompt, "Individual Asset Output", variants_str
        ])

    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    preview_out = saved_paths[0]
    return preview_out, saved_paths, warning_msg, preview_out

def on_gallery_select(evt: gr.SelectData, current_images):
    if not current_images or evt.index is None:
        return None, "", None

    try:
        selected_idx = evt.index
        selected_item = current_images[selected_idx]

        if isinstance(selected_item, dict):
            selected_image_source = selected_item.get("name", "")
        elif isinstance(selected_item, (list, tuple)):
            selected_image_source = selected_item[0]
        else:
            selected_image_source = str(selected_item)

        return selected_image_source, "", selected_image_source

    except Exception as e:
        return None, f'<div style="color: #f87171;">Selection parser error: {str(e)}</div>', None

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

    return save_path, '<div style="color: #4ade80; font-weight: bold; margin-top: 5px;">✨ Modification applied successfully!</div>'

def reset_to_original_image(backup_path):
    if not backup_path:
        return gr.update(), '<div style="color: #fbbf24;">No original image frame cached. Click a gallery target first!</div>'
    return backup_path, '<div style="color: #60a5fa;">🔄 Reverted back to the original layout frame.</div>'

def append_to_favorites(target_img, custom_name, current_favorites):
    if not target_img:
        return current_favorites, gr.update(), '<div style="color: #f87171;">Generate or click an image before saving.</div>'

    try:
        if isinstance(target_img, str):
            base_image_pil = Image.open(target_img)
        else:
            base_image_pil = Image.fromarray(target_img)

        if custom_name and custom_name.strip():
            clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', custom_name.strip())
            display_title = clean_name
            filename = f"{clean_name}.png"
        else:
            display_title = f"Asset_{datetime.now().strftime('%M%S')}"
            filename = f"fav_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.png"

        source_path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
        base_image_pil.save(source_path)

        updated_favorites = list(current_favorites)
        existing_paths = [item[0] for item in updated_favorites if isinstance(item, (tuple, list))]

        if source_path not in existing_paths:
            updated_favorites.append((source_path, display_title))
            status = f'<div style="color: #4ade80; font-weight: bold; margin-top: 5px;">💖 Saved successfully as "{display_title}"! Check the Saved Images tab.</div>'
        else:
            status = '<div style="color: #fbbf24; margin-top: 5px;">An image with this identical storage filename already exists.</div>'

    except Exception as e:
        return current_favorites, gr.update(), f'<div style="color: #f87171;">Failed saving: {str(e)}</div>'

    return updated_favorites, gr.update(value=updated_favorites), status

def set_processing_notice():
    return '<div style="text-align: center; width: 100%; font-weight: bold; color: #fbbf24; padding: 0; margin: 10px 0;">⏳ Processing Pipeline Initiated...</div>'

def clear_workspace_preview():
    return None

GENERATOR_CSS = """
html, body, grad-app, .gradio-container {
    background: 
        radial-gradient(circle at 15% 50%, rgba(236, 72, 153, 0.40), transparent 50%),
        radial-gradient(circle at 85% 30%, rgba(56, 189, 248, 0.40), transparent 50%),
        radial-gradient(circle at 50% 90%, rgba(139, 92, 246, 0.40), transparent 50%),
        linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #020617 100%) !important;
    background-attachment: fixed !important;
    min-height: 100vh !important;
}

.gradio-container {
    padding: 30px !important;
}

/* 1. STRUCTURAL CLEANUP - ELIMINATES STACKED BACKDROP LAYERS */
.gradio-container .block,
.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container .group,
.gradio-container .gr-group,
.gradio-container div[class*="svelte-"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Force Equal Columns and Vertical Stretching */
.gradio-container .row, 
.gradio-container div[class*="row"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    display: flex !important;
    align-items: stretch !important;  /* Stretches side-by-side components equally down */
}

.gradio-container .column,
.gradio-container div[class*="column"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Sketch to Image spacing and structural border recovery */
.sketch-upload-wrapper {
    margin-left: 24px !important;  /* Creates the separation gap from the left board */
}

.sketch-upload-wrapper .image-container {
    border: 2px dashed rgba(255, 255, 255, 0.25) !important; /* Re-establishes a clean, noticeable border */
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.03) !important;
}

/* 2. BRIGHTER GLASSFIELDS */
.gradio-container input, 
.gradio-container textarea, 
.gradio-container select,
.gradio-container div[class*="token-input"],
.gradio-container .tabitem,
.gradio-container fieldset,
.gradio-container .box,
.gradio-container div[class*="input"] {
    background-color: rgba(255, 255, 255, 0.25) !important;
    background: rgba(255, 255, 255, 0.25) !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

.gradio-container label[class*="wrapper"], 
.gradio-container .form .dark {
    background-color: rgba(255, 255, 255, 0.14) !important;
    background: rgba(255, 255, 255, 0.14) !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
}

.gradio-container input:focus, 
.gradio-container textarea:focus {
    border-color: rgba(236, 72, 153, 0.7) !important;
    background-color: rgba(255, 255, 255, 0.22) !important;
}

/* 3. FLUID SINGLE-PANEL MAIN CARDS WITH RESTORED TAB SPACING */
.control-settings-card, .modify-panel-card {
    background: rgba(255, 255, 255, 0.05) !important; 
    backdrop-filter: blur(24px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(160%) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 24px !important;
    box-shadow: 0 30px 60px 0 rgba(0, 0, 0, 0.3) !important;
    padding: 24px !important;
    margin-top: 35px !important;     /* FIXED: Added a robust top margin to isolate from headers */
    margin-bottom: 20px !important;
    flex-grow: 1 !important;  /* Pull structural box sizes cleanly to fill the gap */
}

/* 4. TARGETING THE EXPLICIT OUTPUT GALLERY CARD CLASS WITH RESTORED TAB SPACING */
.gradio-container .output-gallery-card {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
    border: 2px dashed rgba(255, 255, 255, 0.25) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    margin-top: 35px !important;       /* FIXED: Generates breathing space between gallery canvas and tab borders */
    margin-bottom: 20px !important; 
    box-shadow: 0 20px 50px 0 rgba(0, 0, 0, 0.3) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    
    /* Forces the box structure itself to stretch down cleanly */
    display: flex !important;
    flex-direction: column !important;
    flex-grow: 1 !important; 
    height: calc(100% - 55px) !important; 
}

/* Forces inner Gradio gallery layers to absorb all newly stretched vertical room */
.gradio-container .output-gallery-card > .grid-wrap,
.gradio-container .output-gallery-card div[class*="gallery"] {
    flex-grow: 1 !important;
    height: 100% !important;
}

.output-gallery-card .upload-container,
.output-gallery-card .grid-wrap div[class*="empty"],
.output-gallery-card div[class*="upload"] {
    display: none !important;
}

.gradio-container .output-gallery-card .preview,
.gradio-container .output-gallery-card div[class*="empty"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* 5. LABELS AND TEXT ACCENTS (UPDATED FOR MAXIMUM CONTRAST) */
.gradio-container label span, 
.gradio-container .text-sm,
.gradio-container p,
.gradio-container span,
.gradio-container label,
.gradio-container .block-title {
    color: #ffffff !important;
    font-weight: 600 !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4) !important;
}

/* 6. PRIMARY INTERACTIVE BUTTONS */
.button-row {
    display: flex !important;
    justify-content: center !important; 
    width: 100% !important;
    margin-top: 15px !important;
}

.gradio-container button.primary, 
.gradio-container button[class*="primary"] {
    background: linear-gradient(90deg, #ec4899, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 14px 36px !important;
    cursor: pointer !important;
    max-width: 260px !important;
    width: 100% !important; 
    display: block !important;
    margin: 20px auto 0 auto !important;
    box-shadow: 0 8px 20px rgba(236, 72, 153, 0.4) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.gradio-container button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 12px 24px rgba(236, 72, 153, 0.55) !important;
}

/* CUSTOM WORKSPACE EDITING BUTTONS */
.apply-btn-style {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}

.apply-btn-style:hover {
    background: rgba(255, 255, 255, 0.25) !important;
    border-color: rgba(236, 72, 153, 0.6) !important;
    box-shadow: 0 4px 15px rgba(236, 72, 153, 0.3) !important;
    transform: translateY(-1px);
}

.reset-btn-style {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

.reset-btn-style:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    color: #ffffff !important;
    border-color: rgba(56, 189, 248, 0.5) !important;
    box-shadow: 0 4px 15px rgba(56, 189, 248, 0.2) !important;
    transform: translateY(-1px);
}

#mode_radio_group {
    background: none !important;
    border: none !important;
}

#mode_radio_group label {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

#mode_radio_group label.selected {
    background: rgba(236, 72, 153, 0.4) !important;
    border: 1px solid rgba(236, 72, 153, 0.8) !important;
    font-weight: bold !important;
}

.fav-matrix-container .grid-container,
.fav-matrix-container .gallery {
    grid-template-columns: repeat(auto-fill, minmax(130px, 130px)) !important;
    gap: 12px !important;
}
.fav-matrix-container .thumbnail-item,
.fav-matrix-container img {
    height: 110px !important;
    width: 130px !important;
    object-fit: cover !important;
    border-radius: 8px !important;
}
.fav-matrix-container .caption-label {
    font-size: 0.85em !important;
    color: #cbd5e1 !important;
    text-align: center !important;
    margin-top: 4px !important;
    word-break: break-all !important;
}

.fav-matrix-container {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    min-height: 500px !important;
    margin-top: 35px !important;       /* FIXED: Gives matching spacing to the Saved Gallery tab */
}

.gallery-column {
    display: flex !important;
    flex-direction: column !important;
}

.button-row { width: 100% !important; margin-top: 10px !important; }

.editing-buttons-row {
    margin-top: 25px !important;
}

.compact-number {
    max-width: 140px !important;
    width: 100% !important;
}

footer { display: none !important; }
"""

def update_ui(mode_selection):
    if mode_selection == "Sketch to Image":
        return gr.update(visible=True), gr.update(visible=True), gr.update(interactive=True, label="Upload your sketch and guide your sketch details")
    elif mode_selection == "Fantasy Images":
        return gr.update(visible=False), gr.update(visible=True), gr.update(interactive=True, label="Unique environmental variations")
    else:
        return gr.update(visible=False), gr.update(visible=True), gr.update(interactive=True, label="Describe the image you want to generate")

def create_generator_ui():
    original_image_backup = gr.State(None)
    favorites_cache = gr.State([])

    gr.Markdown("# 🎨 AI Multimode Image Studio")

    with gr.Tabs():
        with gr.TabItem("Studio Workspace"):
            with gr.Row():
                with gr.Column(scale=3, elem_classes=["gallery-column"]):
                    output_gallery = gr.Gallery(show_label=False, columns=5, height="auto", type="filepath", elem_classes=["output-gallery-card"], interactive=False)

                with gr.Column(scale=3):
                    with gr.Group(visible=True, elem_classes=["modify-panel-card"]) as modify_panel:
                        gr.Markdown("### 🛠️ Workspace Editing Panel")
                        selected_preview = gr.Image(show_label=False, type="filepath", interactive=False)

                        modify_input_prompt = gr.Textbox(label="Prompt Modification", placeholder="Describe adjustments... e.g., 'wearing a red collar'")
                        strength_control = gr.Number(minimum=0.10, maximum=0.90, value=0.45, label="Transformation Strength", info = "Lower values retain more of the original image; higher values apply more of the new prompt's influence.")

                        with gr.Row():
                            with gr.Row(elem_classes=["editing-buttons-row"]):
                                submit_modification_btn = gr.Button("Apply Changes", variant="secondary", size="sm", elem_classes=["apply-btn-style"])
                                reset_original_btn = gr.Button("🔄 Reset to Original", variant="secondary", size="sm", elem_classes=["reset-btn-style"])
                        gr.Markdown("---")
                        custom_filename_input = gr.Textbox(label="Save As", placeholder="e.g., cyber_dog_neon", lines=1)
                        save_favorite_btn = gr.Button("💖 Save to Gallery", variant="primary", size="md")
                        modification_status = gr.HTML("")

            # Generation Controls Board
            with gr.Group(elem_classes="control-settings-card"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            with gr.Column(scale=2):
                                mode = gr.Radio(choices=["Text to Image", "Sketch to Image", "Fantasy Images"], value="Text to Image", label="1. Choose Your Generation Mode", interactive=True, elem_id = "mode_radio_group")
                            with gr.Column(scale=1):
                                count_slider = gr.Number(value=1, minimum=1, maximum=100, precision=0, label="2. Style Variations", elem_classes=["compact-number"])

                        with gr.Row():
                            prompt = gr.Textbox(value="", label="Prompt (Describe the image you want to generate)", lines=3)

                        generate_btn = gr.Button("Generate Image", variant="primary", elem_classes=["button-row"])

                    with gr.Column(scale=1, min_width=250, elem_classes=["sketch-upload-wrapper"]):
                        with gr.Group(visible=False) as sketch_inputs:
                            sketch_img = gr.Image(type="pil", show_label=False, sources=["upload", "clipboard"], height=250)

                status_message = gr.HTML("")

        with gr.TabItem("Saved Gallery"):
            with gr.Group(elem_classes=["fav-matrix-container"]):
                gr.Markdown("### 🌟 Saved Artifact Grid")
                saved_gallery = gr.Gallery(label="Archived Asset Scrapbook", columns=10, type="filepath", height="auto", object_fit="cover", preview=False, show_label=False)
                
    return {
        "mode": mode,
        "count_slider": count_slider,
        "sketch_inputs": sketch_inputs,
        "sketch_img": sketch_img,
        "prompt": prompt,
        "generate_btn": generate_btn,
        "selected_preview": selected_preview,
        "output_gallery": output_gallery,
        "status_message": status_message,
        "original_image_backup": original_image_backup,
        "modify_input_prompt": modify_input_prompt,
        "strength_control": strength_control,
        "submit_modification_btn": submit_modification_btn,
        "reset_original_btn": reset_original_btn,
        "custom_filename_input": custom_filename_input,
        "save_favorite_btn": save_favorite_btn,
        "modification_status": modification_status,
        "favorites_cache": favorites_cache,
        "saved_gallery": saved_gallery
    }

with gr.Blocks(css=GENERATOR_CSS) as demo:
    ui = create_generator_ui()
    
    # --- Event Wiring ---
    ui["mode"].change(fn=update_ui, inputs=[ui["mode"]], outputs=[ui["sketch_inputs"], ui["count_slider"], ui["prompt"]])
    
    ui["generate_btn"].click(
        fn=clear_workspace_preview,
        inputs=None,
        outputs=[ui["selected_preview"]]
    ).then(
        fn=set_processing_notice,
        inputs=None,
        outputs=[ui["status_message"]]
    ).then(
        fn=generate,
        inputs=[ui["mode"], ui["count_slider"], ui["sketch_img"], ui["prompt"]],
        outputs=[ui["selected_preview"], ui["output_gallery"], ui["status_message"], ui["original_image_backup"]]
    )

    ui["output_gallery"].select(
        fn=on_gallery_select,
        inputs=[ui["output_gallery"]],
        outputs=[ui["selected_preview"], ui["modification_status"], ui["original_image_backup"]]
    )

    ui["submit_modification_btn"].click(
        fn=modify_selected_image,
        inputs=[ui["selected_preview"], ui["modify_input_prompt"], ui["strength_control"], ui["prompt"]],
        outputs=[ui["selected_preview"], ui["modification_status"]]
    )

    ui["reset_original_btn"].click(
        fn=reset_to_original_image,
        inputs=[ui["original_image_backup"]],
        outputs=[ui["selected_preview"], ui["modification_status"]]
    )

    ui["save_favorite_btn"].click(
        fn=append_to_favorites,
        inputs=[ui["selected_preview"], ui["custom_filename_input"], ui["favorites_cache"]],
        outputs=[ui["favorites_cache"], ui["saved_gallery"], ui["modification_status"]]
    )
    
    # --- FORCE DARK MODE ON INITIAL DASHBOARD LOAD ---
    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { document.documentElement.classList.add('dark'); }"
    )

if __name__ == "__main__":
    demo.queue().launch()