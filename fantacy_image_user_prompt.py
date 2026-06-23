import os
import torch
import gc
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import gradio as gr

# --- STEP 1: CONFIGURE ENVIRONMENT & PIPELINE ---
print("\n[1/3] Setting up hardware configurations...")

model_id = "cagliostrolab/animagine-xl-3.0" 

# Initialize base text-to-image pipeline using float16 precision
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,  
    use_safetensors=True
)

# REUSE underlying components for the Img2Img pipeline instead of creating a second model copy
img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)

# Aggressive memory saving guards
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()        # Sequentially offloads layers to system RAM when idle
img2img_pipe.enable_attention_slicing()
img2img_pipe.enable_model_cpu_offload()

print("Model successfully optimized for Colab environments.\n")

# Setup targeted file storage paths
os.makedirs("personalized_avatars", exist_ok=True)
current_avatar_image = None


# --- STEP 2: ENGINE PROCESSING CORDS ---
def generate_initial_avatar(gender, race, age, hair, eyes, expression, clothing, accessories, palette, art_style, background):
    global current_avatar_image
    
    # Structural prompt aggregation
    final_prompt = (
        f"masterpiece, best quality, {art_style} portrait, 1person, "
        f"{age} {gender} {race}, {hair}, {eyes}, {expression}, "
        f"wearing {clothing}, {accessories}, color palette of {palette}, "
        f"{background}, highly detailed, 8k"
    )

    negative_prompt = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
        "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
        "username, blurry, bad feet, out of frame, full body, multiple people"
    )
    
    print(f"\nProcessing Text-to-Image Generation...")
    
    with torch.inference_mode():
        image = pipe(
            prompt=final_prompt, 
            negative_prompt=negative_prompt, 
            num_inference_steps=28,     # Reduced steps for stability on standard Colab GPU
            guidance_scale=7.0,
            height=1024,  
            width=1024
        ).images[0]
        
    save_path = "personalized_avatars/custom_avatar.png"
    image.save(save_path)
    
    current_avatar_image = image  
    
    # Mandatory aggressive memory sweep
    gc.collect()
    torch.cuda.empty_cache()
    
    return image, gr.update(visible=True)


def modify_existing_avatar(modification_text):
    global current_avatar_image
    
    if current_avatar_image is None:
        raise gr.Error("Please generate an initial avatar first before requesting changes!")
        
    print(f"\nProcessing Img2Img modification request...")
    
    refinement_prompt = f"masterpiece, best quality, highly detailed, {modification_text}"
    negative_prompt = "lowres, bad anatomy, worst quality, low quality, blurry, text, watermark"
    
    with torch.inference_mode():
        updated_image = img2img_pipe(
            prompt=refinement_prompt,
            negative_prompt=negative_prompt,
            image=current_avatar_image,
            strength=0.35,              # 0.35 preserves core identity layouts during modification
            guidance_scale=7.5,
            num_inference_steps=20      # Low steps keep image-to-image adjustments light on RAM
        ).images[0]
        
    save_path = "personalized_avatars/custom_avatar_modified.png"
    updated_image.save(save_path)
    
    current_avatar_image = updated_image  
    
    # Mandatory aggressive memory sweep
    gc.collect()
    torch.cuda.empty_cache()
    
    return updated_image


# --- STEP 3: INTERACTIVE GRADIO INTERFACE LAYOUT ---
with gr.Blocks() as demo:
    gr.Markdown("# 👤 Personalized Single-Avatar Studio")
    gr.Markdown("Fill out your character's traits below to generate a custom high-resolution avatar, then request targeted edits.")
    
    with gr.Row():
        # Left Panel: Base Setup
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Character Trait Form")
            gender = gr.Textbox(label="Gender / Presentation", value="Feminine elf")
            race = gr.Textbox(label="Race / Species", value="Cyborg")
            age = gr.Textbox(label="Apparent Age", value="Young adult")
            hair = gr.Textbox(label="Hair Style & Color", value="Short messy silver-white hair")
            eyes = gr.Textbox(label="Eye Details", value="One glowing blue cybernetic eye, one purple eye")
            expression = gr.Textbox(label="Expression / Facial Features", value="Confident smirk, sleek facial tech lines")
            clothing = gr.Textbox(label="Attire / Clothing", value="Sleek dark sci-fi armor suit")
            accessories = gr.Textbox(label="Accessories", value="Cybernetic headpiece")
            palette = gr.Textbox(label="Color Palette", value="Neon cyan, purple, and dark metallic gray")
            art_style = gr.Dropdown(
                choices=["Digital illustration", "Realistic portrait", "Stylized anime", "Cyberpunk concept art", "Watercolor"], 
                label="Art Style", 
                value="Stylized anime"
            )
            background = gr.Textbox(label="Background Setting", value="Blurry cyberpunk city streets with neon signs")
            
            generate_btn = gr.Button("🔮 Generate Baseline Avatar", variant="primary")
            
        # Right Panel: Output & Iterative Edits
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Your Custom Avatar")
            output_viewport = gr.Image(label="Generated Result (1024x1024)")
            
            # Using gr.Group instead of deprecated gr.Box
            with gr.Group(visible=False) as modification_panel:
                gr.Markdown("### 🪄 Request Modifications")
                change_input = gr.Textbox(
                    label="What changes would you like to make?", 
                    placeholder="e.g., 'Make her hair long and flowing' or 'Change background to a sunny forest'"
                )
                modify_btn = gr.Button("✨ Apply Changes to Avatar", variant="secondary")

    # Wire up button event triggers
    generate_btn.click(
        fn=generate_initial_avatar,
        inputs=[gender, race, age, hair, eyes, expression, clothing, accessories, palette, art_style, background],
        outputs=[output_viewport, modification_panel]
    )
    
    modify_btn.click(
        fn=modify_existing_avatar,
        inputs=[change_input],
        outputs=[output_viewport]
    )

# Launch with share=True to bypass Google Colab proxy websocket bottlenecks
demo.launch(share=True, debug=True)
