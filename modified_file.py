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
    warning_msg = '<div style="text-align: center; width: 100%; font-size: 1.1em; color: #fff; background: transparent; border: none; padding: 0; margin: 10px 0;">⚠ AI can make mistakes. Please verify important outputs.</div>'

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

    return save_path, '<div style="color: #4ade80; font-weight: bold;">✨ Workspace modifications applied! Check your target window.</div>'


def reset_workspace_image(backup_path):
    if not backup_path:
        return gr.update(), '<div style="color: #f87171;">No original image available to reset to.</div>'
    return backup_path, '<div style="color: #60a5fa;">🔄 Reset workspace back to your original base image. Try altering strength values!</div>'


def set_processing_notice():
    return (
        '<div style="text-align: center; width: 100%; font-weight: bold; color: #fbbf24; background: transparent; border: none; padding: 0; margin: 10px 0;">⏳ Processing Pipeline Initiated... <span style="font-weight: normal; font-size: 0.9em; color: #fff;">(AI can make mistakes. Generating asset frames...)</span></div>',
        gr.update(visible=False)
    )


def append_to_favorites(target_img, custom_name, current_favorites):
    if not target_img:
        return current_favorites, gr.update(), '<div style="color: #f87171;">Select or modify an image before archiving.</div>'
    
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
            status = f'<div style="color: #4ade80; font-weight: bold;">💖 Saved as "{clean_name}" into Favorites Scrapbook!</div>'
        else:
            status = f'<div style="color: #fbbf24;">Image is already archived under this configuration pathway.</div>'
            
    except Exception as e:
        return current_favorites, gr.update(), f'<div style="color: #f87171;">Failed saving file asset: {str(e)}</div>'
        
    return current_favorites, gr.update(value=current_favorites, visible=True), status


def filter_favorites(query, current_favorites):
    if not query.strip():
        return gr.update(value=current_favorites)
    
    q = query.lower().strip()
    filtered = [path for path in current_favorites if q in os.path.basename(path).lower()]
    return gr.update(value=filtered)


# --- 6. GRADIO INTERFACE CONFIGURATION & CUSTOM STYLE RULES ---
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

/* Ensure raw rows/columns don't carry accidental background styles */
.gradio-container .row, 
.gradio-container .column,
.gradio-container div[class*="row"],
.gradio-container div[class*="column"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* 2. BRIGHTER GLASSFIELDS - REMOVES HARSH DARK TEXT BOX BLOCKS */
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

/* Brighten radio input choices blocks */
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

/* 3. FLUID SINGLE-PANEL MAIN CARDS (BLURS TO GRADIENT SMOOTHLY) */
.control-settings-card, .modify-panel-card {
    background: rgba(255, 255, 255, 0.05) !important; 
    backdrop-filter: blur(24px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(160%) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 24px !important;
    box-shadow: 0 30px 60px 0 rgba(0, 0, 0, 0.3) !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
}

/* 4. TARGETING THE EXPLICIT OUTPUT GALLERY CARD CLASS */
.gradio-container .output-gallery-card {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
    border: 2px dashed rgba(255, 255, 255, 0.25) !important; /* <--- THIS LINE ADDS THE BORDER */
    border-radius: 20px !important;
    padding: 24px !important;
    margin-bottom: 25px !important;
    min-height: 440px !important;
    box-shadow: 0 20px 50px 0 rgba(0, 0, 0, 0.3) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}

.gradio-container .output-gallery-card .preview,
.gradio-container .output-gallery-card div[class*="empty"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* 5. LABELS AND TEXT ACCENTS */
.gradio-container label span, 
.gradio-container .text-sm,
.gradio-container p {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
}

/* 6. PRIMARY INTERACTIVE BUTTON */
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
    margin: 0 auto !important; 
    box-shadow: 0 8px 20px rgba(236, 72, 153, 0.4) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.gradio-container button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 12px 24px rgba(236, 72, 153, 0.55) !important;
}

/* Update these rules to specifically target the children of your radio group */

/* 1. Style the container itself if needed */
#mode_radio_group {
    background: none !important;
    border: none !important;
    border-radius: none !important;
    padding: none !important;
}

/* 2. Style the individual label wrappers inside your group */
#mode_radio_group label {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

/* 3. Style the selected state specifically */
#mode_radio_group label.selected {
    background: rgba(236, 72, 153, 0.4) !important;
    border: 1px solid rgba(236, 72, 153, 0.8) !important;
    font-weight: bold !important;
}
footer { display: none !important; }
"""

def update_ui(mode_selection):
    if mode_selection == "Sketch to Image":
        return (
            gr.update(visible=True),      # sketch_inputs
            gr.update(visible=True),      # count_slider
            gr.update(interactive=True, label="Prompt (Guide your sketch details)")
        )
    elif mode_selection == "Fantasy Images":
        return (
            gr.update(visible=False),
            gr.update(visible=True),      # keep counter visible
            gr.update(interactive=True, label="Core Idea (Unique environmental variations will match this)")
        )
    else:  # Text to Image
        return (
            gr.update(visible=False),
            gr.update(visible=True),      # keep counter visible
            gr.update(interactive=True, label="Prompt (Describe the image you want to generate)")
        )

def create_generator_ui():
    original_image_backup = gr.State(None)
    favorites_cache = gr.State([]) 
    
    gr.Markdown("# 🎨 AI Multimode Image Studio")

    with gr.Tabs():
        with gr.TabItem("Studio Workspace"):
            with gr.Row():
                with gr.Column(scale=4):
                    #processed_preview = gr.Image(label="Processed Edge Map Preview", type="filepath", visible=False)
                    output_gallery = gr.Gallery(label="Generated Output Images", columns=4, height="auto", type="filepath", elem_classes=["output-gallery-card"])
                
                with gr.Column(scale=2):
                    with gr.Group(visible=False, elem_classes=["modify-panel-card"]) as modify_panel:
                        gr.Markdown("### 🛠️ Modify Workspace")
                        selected_preview = gr.Image(label="Target Image", type="filepath", interactive=False)
                        with gr.Row():
                            reset_btn = gr.Button("🔄 Revert", size="sm", variant="secondary")
                            save_favorite_btn = gr.Button("💖 Save", size="sm", variant="primary")
                        custom_filename_input = gr.Textbox(label="Custom Name", placeholder="filename...", max_lines=1)
                        modify_input_prompt = gr.Textbox(label="Modifications", placeholder="e.g., 'deep red water'")
                        strength_control = gr.Slider(minimum=0.10, maximum=0.90, value=0.45, step=0.05, label="Strength")
                        submit_modification_btn = gr.Button("Apply", variant="secondary")
                        modification_status = gr.HTML("")

            # Bottom Controls Card
            with gr.Group(elem_classes="control-settings-card"):
                with gr.Row():
                    # Right side column
                    with gr.Column(scale=2):
                        with gr.Row():
                            with gr.Column(scale=2):
                                mode = gr.Radio(choices=["Text to Image", "Sketch to Image", "Fantasy Images"], value="Text to Image", label="1. Choose Your Generation Mode", elem_id="mode_radio_group", interactive=True)
                            with gr.Column(scale=1, min_width=150):
                                count_slider = gr.Number(value=1, minimum=1, maximum=100, precision=0, label="2. Number of Style Variations", interactive=True, elem_id="short_counter_box")
                       
                        # Row containing prompt
                        with gr.Row():
                            with gr.Column(scale=3):
                                prompt = gr.Textbox(value="", label="Prompt (Describe the image you want to generate)", lines=4)
                        
                        # Row containing generate button
                        generate_btn = gr.Button(
                                    "Generate Image", 
                                    variant="primary", 
                                    elem_classes=["button-row"], # Your existing shared class
                                    elem_id="generate-btn-id"    # The new unique ID for centering
                                )
                             # Left side column
                    with gr.Column(scale=1, min_width=300):
                        with gr.Group(visible=False) as sketch_inputs:
                            sketch_img = gr.Image(type="pil", label="Upload or Draw Sketch", sources=["upload", "clipboard"], height=320, elem_id="sketch_input_box")
                       
                status_message = gr.HTML("")

        with gr.TabItem("Saved Favorites Scrapbook"):
        
            gr.Markdown("### 🌟 Saved Favorites Gallery")
            search_bar = gr.Textbox(label="🔍 Filter Favorites by Name", placeholder="Type to search...", max_lines=1)
            saved_gallery = gr.Gallery(elem_id="saved_favorites_gallery", label="Your Collected Artifact Scrapbook", columns=6, type="filepath", visible=False)

    return {
        "output_gallery": output_gallery, "modify_panel": modify_panel, 
        "selected_preview": selected_preview, "modify_input_prompt": modify_input_prompt, "strength_control": strength_control, 
        "submit_modification_btn": submit_modification_btn, "modification_status": modification_status, "mode": mode, 
        "count_slider": count_slider, "prompt": prompt, "sketch_inputs": sketch_inputs, "sketch_img": sketch_img, 
        "generate_btn": generate_btn, "status_message": status_message, "search_bar": search_bar, "saved_gallery": saved_gallery, 
        "reset_btn": reset_btn, "save_favorite_btn": save_favorite_btn, "custom_filename_input": custom_filename_input,
        "original_image_backup": original_image_backup, "favorites_cache": favorites_cache
    }

if __name__ == "__main__":
    with gr.Blocks(css=GENERATOR_CSS) as demo:
        ui = create_generator_ui()
        ui["search_bar"].change(fn=filter_favorites, inputs=[ui["search_bar"], ui["favorites_cache"]], outputs=[ui["saved_gallery"]])
        ui["mode"].change(fn=update_ui, inputs=ui["mode"], outputs=[ui["sketch_inputs"], ui["count_slider"], ui["prompt"]])
        ui["generate_btn"].click(fn=set_processing_notice, inputs=None, outputs=[ui["status_message"], ui["modify_panel"]]).then(
            fn=generate, inputs=[ui["mode"], ui["count_slider"], ui["sketch_img"], ui["prompt"]], outputs=[ui["output_gallery"], ui["status_message"], gr.State([])]
        )
        
        # --- FORCE DARK MODE ON INITIAL DASHBOARD LOAD ---
        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { document.documentElement.classList.add('dark'); }"
        )

    demo.queue().launch()