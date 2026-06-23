import os
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# -------------------------------------------------------------------------
# 1. Pipeline Initialization
# -------------------------------------------------------------------------
print("\nLoading the Stable Diffusion model...")
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
)
pipe = pipe.to("cuda")
print("Model loaded successfully.\n")

# 2. Capture the Prompt Concept
print("\n--- Custom 10-Variant Fantasy Generator ---")
user_idea = input("What kind of fantasy scene do you want to create? (e.g., A wizard tower, A cyber-dragon): ")

styles = [
    "cinematic lighting, floating on a cloud island, waterfalls cascading into the sky, highly detailed, 8k resolution, digital art masterpiece",
    "shimmering aurora borealis sky, iridescent crystal structures, ethereal magical glow, surreal painting, high fantasy, hyper-detailed",
    "dramatic thunderstorm, jagged crackling lightning, dark epic atmosphere, high contrast, photo-realistic rendering, highly detailed",
    "golden hour sunset, warm god rays, flying dust particles, majestic mood, volumetric lighting, oil painting style, artstation trending",
    "mystical star-filled night sky, giant full moon, glowing liquid neon waterfalls, cosmic fantasy concept art, highly intricate",
    "steampunk style, complex brass gears, winding copper pipes, churning thick steam clouds, copper sunset lighting, highly detailed",
    "ancient overgrown ruins, glowing emerald moss, dense jungle vines, sun-dappled rainforest canopy, photorealistic nature fantasy",
    "frozen winter blizzard, floating glacial mountain peaks, clear icicles, cool blue-toned lighting, crisp digital illustration",
    "surreal desert oasis, dry sand waterfalls, harsh midday sun, blazing horizon, heat haze effect, gritty fantasy style",
    "cyberpunk magic hybrid, glowing holographic runes, deep purple and cyan neon haze, dark moody atmosphere, futuristic fantasy"
]

negative_prompt = "ugly, deformed, blurry, modern buildings, cars, low quality, text, watermark, bad anatomy, rectangular, wide"

# Set up local directory storage structures
os.makedirs("fantasy_variants", exist_ok=True)
os.makedirs("fantasy_compositions", exist_ok=True)

safe_base = "".join(c for c in user_idea if c.isalnum() or c in (' ', '_', '-')).rstrip()
safe_base = safe_base.replace(" ", "_").lower()[:20]

print(f"\nGenerating 10 unique variants of your idea: '{user_idea}'...\n")
generated_files = []

# 3. Mass Generation Stage
for i, style in enumerate(styles, start=1):
    prompt = f"{user_idea}, {style}"
    print(f"--- Generating Image {i}/10 ---")
    
    with torch.inference_mode():
        image = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            num_inference_steps=30, 
            guidance_scale=7.5,
            height=512,  
            width=512    
        ).images[0]
    
    filename = f"fantasy_variants/{safe_base}_variant_{i}.png"
    image.save(filename)
    generated_files.append(filename)
    print(f"Saved: '{filename}'")
    torch.cuda.empty_cache()

print("\nAll 10 images generated! Launching the Interactive Mixing & Layering Dashboard below...\n")


# -------------------------------------------------------------------------
# 4. Core Layer Mixing Logic Engine
# -------------------------------------------------------------------------
def create_composition(bg_idx, crop_output_img, x_pos, y_pos, opacity, text_input, txt_x, txt_y, txt_color):
    """
    Combines background images, visually cropped element layers, and typography into a new asset.
    """
    if bg_idx is None:
        return None
    
    # 1. Grab chosen background file
    bg_path = generated_files[int(bg_idx)]
    bg_img = Image.open(bg_path).convert("RGBA")
    
    # 2. Setup the final composition canvas
    composition_canvas = Image.new("RGBA", bg_img.size)
    composition_canvas.paste(bg_img, (0, 0))
    
    # 3. If the user cropped an element, process it and overlay it
    if crop_output_img is not None:
        # Convert the cropped numpy array/PIL image from Gradio to RGBA
        if isinstance(crop_output_img, dict) and "background" in crop_output_img:
             fg_element = crop_output_img["background"].convert("RGBA")
        else:
             fg_element = crop_output_img.convert("RGBA")
        
        # Apply alpha opacity modifications to the floating layer if requested
        if opacity < 1.0:
            alpha = fg_element.split()[3]
            alpha = alpha.point(lambda p: int(p * opacity))
            fg_element.putalpha(alpha)
            
        # Paste the cropped element onto the canvas at user coordinates
        composition_canvas.paste(fg_element, (int(x_pos), int(y_pos)), fg_element)
    
    # 4. Bake Custom Typography text layers on top
    if text_input.strip():
        draw = ImageDraw.Draw(composition_canvas)
        try:
            font = ImageFont.load_default(size=28)
        except:
            font = ImageFont.load_default()
        
        draw.text((int(txt_x), int(txt_y)), text_input, fill=txt_color, font=font)
        
    # Finalize format structure from alpha layers to strict standard RGB for viewability
    final_output = composition_canvas.convert("RGB")
    save_path = "fantasy_compositions/custom_masterpiece.png"
    final_output.save(save_path)
    return final_output


# -------------------------------------------------------------------------
# 5. Interface Layout Construction
# -------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🌌 Multiverse Fantasy Collage Studio")
    gr.Markdown("Pick assets from your generated pool, crop elements right on screen, and position them onto a new background canvas!")
    
    with gr.Row():
        # Column 1: Asset Pool & Background Selection
        with gr.Column(scale=1):
            gr.Markdown("### 📜 1. Asset Reference Pool")
            gr.Gallery(value=generated_files, columns=2, label="Generated Pool Gallery")
            
            bg_selector = gr.Dropdown(
                choices=[(f"Variant {i}", str(i-1)) for i in range(1, 11)], 
                label="Select Base Background Canvas", 
                value="0"
            )
            
            fg_selector = gr.Dropdown(
                choices=[(f"Variant {i}", str(i-1)) for i in range(1, 11)], 
                label="Select Element Layer Source", 
                value="1"
            )
            
        # Column 2: Crop Tool & Placement Controls
        with gr.Column(scale=1.5):
            gr.Markdown("### ✂️ 2. Visual Cropping Window")
            gr.Markdown("*Click the crop icon tool inside the box below to isolate your element, then press apply inside the tool.*")
            
            # This image editor dynamically switches when the user changes their dropdown selection
            crop_canvas = gr.ImageEditor(
                type="pil",
                crop_size=(1,1),
                transforms=["crop"], 
                label="Crop Area Workspace"
            )
            
            gr.Markdown("### 🎛️ 3. Layer Placement Controls")
            with gr.Row():
                x_coord = gr.Slider(minimum=-100, maximum=512, value=50, step=5, label="Placement Coordinate (X)")
                y_coord = gr.Slider(minimum=-100, maximum=512, value=50, step=5, label="Placement Coordinate (Y)")
                
            layer_opacity = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Element Layer Opacity")
            
            gr.Markdown("### 🔤 4. Typography Overlay Layer")
            text_str = gr.Textbox(label="Text Content", placeholder="Type overlay title...")
            with gr.Row():
                t_x = gr.Number(value=30, label="Text Pos X")
                t_y = gr.Number(value=430, label="Text Pos Y")
                t_color = gr.ColorPicker(value="#FFFFFF", label="Text Color")
                
            mix_btn = gr.Button("🔮 Render Composite Masterpiece", variant="primary")
            
        # Column 3: Canvas Viewport Rendering
        with gr.Column(scale=1.5):
            gr.Markdown(". Composite Output Canvas")
            masterpiece_viewer = gr.Image(label="Live Render Output")
            
    # Connect UI Interaction: Automatically update the cropping window when dropdown changes
    def update_crop_source(idx):
        return generated_files[int(idx)]
        
    fg_selector.change(fn=update_crop_source, inputs=fg_selector, outputs=crop_canvas)
    
    # Initialize the crop canvas with the default variant 2 image path
    demo.load(fn=lambda: generated_files[1], outputs=crop_canvas)

    # Connect Final Composition Render Engine Event
    mix_btn.click(
        fn=create_composition, 
        inputs=[
            bg_selector, crop_canvas, 
            x_coord, y_coord, layer_opacity, 
            text_str, t_x, t_y, t_color
        ], 
        outputs=masterpiece_viewer
    )

# Run internal web server instance inside notebooks
demo.launch(inline=True, share=False)
