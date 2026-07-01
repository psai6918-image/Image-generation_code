# pages/dashboard.py
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

# --- DYNAMIC CSS LOADING ---
css_path = os.path.join("assets", "dashboard-layout.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        GENERATOR_CSS = f.read()
else:
    GENERATOR_CSS = ""

# --- DIRECTORY SETUP ---
OUTPUT_DIR = "fantasy_variants"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAVORITES_ROOT = os.path.join(OUTPUT_DIR, "favorites")
os.makedirs(FAVORITES_ROOT, exist_ok=True)


def get_user_favorites_dir(username):
    """
    Dynamically isolates directory paths per user context 
    to prevent cross-user data exposure.
    """
    safe_username = "".join(c for c in str(username) if c.isalnum() or c in ('_', '-')).rstrip()
    if not safe_username:
        safe_username = "anonymous"
    user_dir = os.path.join(FAVORITES_ROOT, safe_username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def load_existing_favorites(username="anonymous"):
    """
    Loads favorites matching the unique identity of the interacting user.
    """
    user_fav_dir = get_user_favorites_dir(username)
    favorites = []
    
    if os.path.exists(user_fav_dir):
        file_list = []
        for f in os.listdir(user_fav_dir):
            if f.endswith(".png"):
                full_path = os.path.join(user_fav_dir, f)
                stat = os.stat(full_path)
                try:
                    creation_time = stat.st_birthtime
                except AttributeError:
                    creation_time = stat.st_ctime
                
                file_list.append((creation_time, os.path.abspath(full_path), f))
        
        file_list.sort(key=lambda x: x[0])
        for _, path, name in file_list:
            display_name = os.path.splitext(name)[0].replace("_", " ")
            favorites.append((path, display_name))
            
    return favorites


# --- MODEL COMPUTATION & HOOKS ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.cuda.empty_cache()

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

# --- PROCEDURAL GENERATIVE ASSETS ---
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


# --- PIPELINE ROUTING FUNCTIONS ---
def generate(mode, count_selection, sketch_img, base_prompt, progress=gr.Progress()):
    start_dt = datetime.now()
    warning_msg = '<div class="theme-notice">Verify important outputs before saving.</div>'

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
        return None, f'<div class="theme-error">Selection parser error: {str(e)}</div>', None


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

    return save_path, '<div class="theme-success">Modification applied successfully!</div>'


def reset_to_original_image(backup_path):
    if not backup_path:
        return gr.update(), '<div class="theme-warning">No original image frame cached. Click a gallery target first!</div>'
    return backup_path, '<div class="theme-info">Reverted back to the original layout frame.</div>'


def append_to_favorites(target_img, custom_name, _, username="anonymous"):
    """
    Saves the target image into the explicit user profile sandbox directory.
    Enforces that file custom names must be unique strictly inside this user's profile.
    """
    user_fav_dir = get_user_favorites_dir(username)

    if not target_img:
        return load_existing_favorites(username), gr.update(), '<div class="theme-error">Nothing to save.</div>'

    prefix = "".join(c for c in custom_name if c.isalnum() or c in (' ', '_', '-')).rstrip() if custom_name else "fav"
    prefix = prefix.replace(" ", "_")

    filename = f"{prefix}.png"
    save_path = os.path.join(user_fav_dir, filename)

    if os.path.exists(save_path):
        display_err_name = prefix.replace("_", " ")
        return load_existing_favorites(username), gr.update(), f'<div class="theme-error">❌ Name already exists! "{display_err_name}" is already taken in your gallery.</div>'

    if isinstance(target_img, str):
        Image.open(target_img).save(save_path)
    else:
        Image.fromarray(target_img).save(save_path)

    updated_favs = load_existing_favorites(username)
    return updated_favs, gr.update(value=updated_favs), '<div class="theme-success">Saved to persistent gallery!</div>'


# --- SAVED GALLERY RENAME LOGIC ---
def on_saved_gallery_select(evt: gr.SelectData, current_favorites):
    """Triggered when an image inside the Saved Gallery is clicked."""
    if not current_favorites or evt.index is None:
        return "", None, ""
    
    try:
        selected_item = current_favorites[evt.index]
        file_path = selected_item[0]
        display_name = selected_item[1]
        
        return display_name, file_path, f'<div class="theme-info">Selected item: "{display_name}". You can now edit its name below.</div>'
    except Exception as e:
        return "", None, f'<div class="theme-error">Selection error: {str(e)}</div>'


def rename_saved_favorite(current_path, new_name, username="anonymous"):
    """Validates uniqueness and renames the file within the user's isolated directory profile."""
    if not current_path or not os.path.exists(current_path):
        return load_existing_favorites(username), '<div class="theme-warning">Please select an image from your saved gallery first!</div>'
    
    if not new_name.strip():
        return load_existing_favorites(username), '<div class="theme-error">New name cannot be empty!</div>'
        
    user_fav_dir = get_user_favorites_dir(username)
    
    clean_new_name = "".join(c for c in new_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    clean_new_name = clean_new_name.replace(" ", "_")
    
    new_filename = f"{clean_new_name}.png"
    new_path = os.path.join(user_fav_dir, new_filename)
    
    if os.path.exists(new_path) and os.path.abspath(new_path) != os.path.abspath(current_path):
        display_err_name = clean_new_name.replace("_", " ")
        return load_existing_favorites(username), f'<div class="theme-error">❌ Name already exists! "{display_err_name}" is already taken in your gallery.</div>'
        
    try:
        os.rename(current_path, new_path)
        updated_favs = load_existing_favorites(username)
        display_success_name = clean_new_name.replace("_", " ")
        return updated_favs, f'<div class="theme-success">Successfully renamed to "{display_success_name}"!</div>'
    except Exception as e:
        return load_existing_favorites(username), f'<div class="theme-error">System error renaming file: {str(e)}</div>'


def set_processing_notice():
    return '<div class="theme-notice processing-pulse">Processing Pipeline Initiated...</div>'


def clear_workspace_preview():
    return None


def update_ui(mode_selection):
    """
    Handles interactive structural UI visibility maps cleanly 
    depending on selected generation configuration context.
    """
    if mode_selection == "Sketch to Image":
        return gr.update(visible=True), gr.update(visible=True), gr.update(interactive=True, label="Upload your sketch and guide your sketch details")
    elif mode_selection == "Fantasy Images":
        return gr.update(visible=False), gr.update(visible=True), gr.update(interactive=True, label="Unique environmental variations")
    else:
        return gr.update(visible=False), gr.update(visible=True), gr.update(interactive=True, label="Describe the image you want to generate")


def create_generator_ui():
    original_image_backup = gr.State(None)
    favorites_cache = gr.State([])
    selected_saved_file_path = gr.State(None)
    
    with gr.Row(elem_classes=["header-navigation-row"]):
        with gr.Column(scale=10):
            gr.Markdown("# AI Multimode Image Studio", elem_id="main-studio-title")
        with gr.Column(scale=2, min_width=180):
            user_menu = gr.Dropdown(
                choices=["Profile Menu", "Account Settings", "Payment", "Log out"],
                value="Profile Menu",
                show_label=False,
                container=False,
                interactive=True,
                elem_classes=["user-menu-btn"]
            )
            
            account_btn = gr.Button("Account", visible=False)
            payment_btn = gr.Button("Payment", visible=False)
            logout_btn = gr.Button("Logout", visible=False)

    tabs = gr.Tabs()
    with tabs:
        with gr.TabItem("Studio Workspace"):
            status_message = gr.HTML("")
            with gr.Row(elem_classes=["workspace-row-layout"]):
                with gr.Column(scale=3, elem_classes=["gallery-column"]):
                    output_gallery = gr.Gallery(show_label=False, columns=5, height="auto", type="filepath", elem_classes=["output-gallery-card"], interactive=False)

                with gr.Column(scale=3, elem_classes=["editing-panel-column"]):
                    with gr.Tabs() as workspace_panes:
                        with gr.TabItem("Generation Controls"):
                            with gr.Group(elem_classes=["control-settings-card"]):
                                with gr.Row(elem_id="top-controls-row"):
                                    with gr.Column(scale=3, min_width=0):
                                        mode = gr.Radio(choices=["Text to Image", "Sketch to Image", "Fantasy Images"], value="Text to Image", label="1. Choose Your Generation Mode", interactive=True, elem_id="mode_radio_group")
                                    with gr.Column(scale=1, min_width=140):
                                        count_slider = gr.Number(value=1, minimum=1, maximum=100, precision=0, label="2. Style Variations", elem_classes=["compact-number"])

                                with gr.Row():
                                    with gr.Column(elem_classes=["shortened-prompt-col"]):
                                        scroll_notice = gr.HTML("")
                                        prompt = gr.Textbox(value="", label="Prompt (Describe the image you want to generate)", lines=3)
                                        generate_btn = gr.Button("Generate Image", variant="primary", elem_classes=["theme-primary-btn"])

                                with gr.Row(elem_classes=["sketch-upload-wrapper"]):
                                    with gr.Group(visible=False) as sketch_inputs:
                                        sketch_img = gr.Image(type="pil", show_label=False, sources=["upload", "clipboard"], height=250)
                        
                        with gr.TabItem("Workspace Editing Panel"):
                            with gr.Group(visible=True, elem_classes=["modify-panel-card"]) as modify_panel:
                                gr.Markdown("### Workspace Editing Panel")
                                selected_preview = gr.Image(show_label=False, type="filepath", interactive=False, elem_classes=["preview-image-box"])

                                modify_input_prompt = gr.Textbox(label="Prompt Modification", placeholder="Describe adjustments... e.g., 'wearing a red collar'")
                                strength_control = gr.Number(minimum=0.10, maximum=0.90, value=0.45, label="Transformation Strength", info="Lower values retain more of the original image.")

                                with gr.Row(elem_classes=["editing-buttons-row"]):
                                    submit_modification_btn = gr.Button("Apply Changes", variant="secondary", size="sm", elem_classes=["apply-btn-style"])
                                    reset_original_btn = gr.Button("Reset to Original", variant="secondary", size="sm", elem_classes=["reset-btn-style"])
                                
                                gr.Markdown("---")
                                custom_filename_input = gr.Textbox(label="Save As", placeholder="e.g., Apple 1", lines=1)
                                save_favorite_btn = gr.Button("Save to Gallery", variant="primary", size="md", elem_classes=["theme-primary-btn"])
                                modification_status = gr.HTML("")

            # --- CORE INTERACTIVE EVENT ROUTER BINDINGS ---
            # This directly binds the mode selector radio button change event to your functional update_ui pipeline wrapper
            mode.change(
                fn=update_ui,
                inputs=[mode],
                outputs=[sketch_inputs, generate_btn, prompt]
            )

        with gr.TabItem("Saved Gallery") as saved_gallery_tab:
            with gr.Row(elem_classes=["saved-gallery-tab-layout"]):
                with gr.Column(scale=4, elem_classes=["saved-gallery-grid-col"]):
                    with gr.Group(elem_classes=["fav-matrix-container"]):
                        saved_gallery = gr.Gallery(
                            show_label=False,
                            columns=5,
                            type="filepath",
                            height="auto",
                            interactive=True,
                            elem_classes=["saved-gallery-component"]
                        )
                
                with gr.Column(scale=2, elem_classes=["saved-gallery-sidebar-col"]):
                    with gr.Group(elem_classes=["modify-panel-card", "renaming-sidebar-card"]):
                        gr.Markdown("### 📝 Manage Saved Image Name")
                        saved_gallery_status = gr.HTML('<div class="sidebar-placeholder">Click an image in your gallery to edit its name.</div>')
                        
                        rename_input_field = gr.Textbox(label="Edit Name", placeholder="Enter new name... e.g., Apple 1")
                        submit_rename_btn = gr.Button("💾 Rename Image", variant="secondary", size="md", elem_classes=["apply-btn-style"])

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
        "saved_gallery": saved_gallery,
        "saved_gallery_tab": saved_gallery_tab,
        "account_btn": account_btn,
        "payment_btn": payment_btn,
        "logout_btn": logout_btn,
        "user_menu": user_menu,
        "saved_gallery_status": saved_gallery_status,
        "rename_input_field": rename_input_field,
        "submit_rename_btn": submit_rename_btn,
        "selected_saved_file_path": selected_saved_file_path
    }