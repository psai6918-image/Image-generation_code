import os
import csv
import torch
import numpy as np
import cv2
import uuid
import secrets
import hashlib
import threading
from datetime import datetime
from PIL import Image
import mysql.connector
from mysql.connector import Error
import gradio as gr
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline

# Speed up matrix multiplication on compatible GPUs
torch.backends.cuda.matmul.allow_tf32 = True

# --- 1. DATABASE CONFIGURATION ---
DB_CONFIG = {
    'host': 'localhost',
    'database': 'dbo',
    'user': 'root',
    'password': 'NameisRoot909'
}

def init_db():
    """Ensures the users table exists in the MySQL database upon startup."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Username VARCHAR(255) NOT NULL UNIQUE,
        Email VARCHAR(255) NOT NULL UNIQUE,
        DOB DATE NULL,
        Password VARCHAR(255) NOT NULL
    );
    """
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute(create_table_query)
            connection.commit()
            print("MySQL Table verified successfully.")
    except Error as e:
        print(f"Error during database initialization: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# --- 2. SECURITY & HASHING FUNCTIONS ---
def hash_password(password: str) -> str:
    """Generates a secure 16-byte random salt and hashes the password using SHA-256."""
    salt = secrets.token_hex(16)
    hashed_bytes = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    return f"{salt}:{hashed_bytes}"

# --- 3. MODEL CONFIGURATION & INITIALIZATION ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_LOG_FILE = "generation_logs.csv"

gpu_lock = threading.Lock()
log_lock = threading.Lock()

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

if device == "cuda":
    pipe_text2img.enable_model_cpu_offload()
    pipe_sketch2img.enable_model_cpu_offload()
else:
    pipe_text2img.enable_attention_slicing()
    pipe_sketch2img.enable_attention_slicing()

print("Models Loaded Successfully!")

# --- 4. IMAGE SLIDER DATA & LOGIC ---
SLIDER_IMAGES = [
    "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=500",
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=500",
    "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?w=500",
    "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=500",
]
images_list = [f"https://picsum.photos/300/300?random={i}" for i in range(1, 21)]

def rotate_slider(current_index):
    next_index = (current_index + 1) % len(SLIDER_IMAGES)
    return SLIDER_IMAGES[next_index], next_index

def preprocess_sketch(pil_image):
    if pil_image is None:
        return None
    img = pil_image.convert('RGB')
    gray_img = img.convert('L')
    np_img = np.array(gray_img)
    edges = cv2.Canny(np_img, 100, 200)
    final_np_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_np_img).resize((512, 512))

# --- 5. GRADIO ACTIONS & ENGINE WORKFLOWS ---
def save_user(username, email, password, confirm_password):
    """Validates properties and registers user. If successful, updates views to target page."""
    username = username.strip()
    email = email.strip()
    
    if not username or not email or not password or not confirm_password:
        return "### ⚠️ All fields are required!", gr.update(), gr.update()
        
    if password != confirm_password:
        return "### ❌ Passwords do not match\nPlease verify both fields.", gr.update(), gr.update()
    
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            
            cursor.execute("SELECT Username FROM users WHERE Username = %s", (username,))
            if cursor.fetchone():
                return f"### ❌ Username '{username}' is already taken.", gr.update(), gr.update()
                
            cursor.execute("SELECT Email FROM users WHERE Email = %s", (email,))
            if cursor.fetchone():
                return "### ❌ This email is already registered.", gr.update(), gr.update()
            
            secure_password_string = hash_password(password)
            insert_query = "INSERT INTO users (Username, Email, Password) VALUES (%s, %s, %s)"
            cursor.execute(insert_query, (username, email, secure_password_string))
            connection.commit()
            
            # SUCCESSFUL NAVIGATION REDIRECT
            success_msg = f"### 🎉 Welcome aboard, {username}!\nRedirecting to studio pipeline..."
            return success_msg, gr.update(visible=False), gr.update(visible=True)
            
    except Error as e:
        return f"### ❌ Database Error\nCould not process request: {e}", gr.update(), gr.update()
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def generate(mode, sketch_img, prompt, num_images, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Please enter a style/content prompt!")

    batch_size = int(num_images)
    processed_sketch = None
    prompts = [prompt] * batch_size
    progress(0, desc="Waiting in queue / Initializing...")

    if mode == "Sketch to Image":
        if sketch_img is None:
            raise gr.Error("Please upload or draw a sketch first!")
        processed_sketch = preprocess_sketch(sketch_img)

    with gpu_lock:
        progress(0.2, desc="Processing batch on GPU...")
        if mode == "Sketch to Image":
            output_images = pipe_sketch2img(
                prompt=prompts,
                image=[processed_sketch] * batch_size,
                controlnet_conditioning_scale=1.0,
                guidance_scale=7.5,
                num_inference_steps=20
            ).images
        else:
            output_images = pipe_text2img(
                prompt=prompts,
                guidance_scale=7.5,
                num_inference_steps=20
            ).images

    saved_filenames = []
    for image in output_images:
        unique_filename = f"generation_{uuid.uuid4().hex}.png"
        save_path = os.path.join(OUTPUT_DIR, unique_filename)
        image.save(save_path)
        saved_filenames.append(unique_filename)

    with log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filenames_str = "; ".join(saved_filenames)
        with open(CSV_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, mode, prompt, batch_size, filenames_str])

    preview = processed_sketch if mode == "Sketch to Image" else gr.update(visible=False)
    return preview, output_images

# --- 6. PREMIUM LUXURY CSS STYLING ---
custom_css = """
.gradio-container {
    background: transparent !important;
    min-height: 100vh;
    padding: 40px 30px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
#dynamic-bg-canvas {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    background: #09090e;
}
.gradio-container h1, .gradio-container h2, .gradio-container h3, 
.gradio-container p, .gradio-container span, .gradio-container label, .gradio-container .prose {
    color: #ffffff !important;
}
.top-nav {
    display: flex; gap: 40px; padding: 10px 0 30px 0; font-size: 13px; font-weight: 700;
    letter-spacing: 2px; color: rgba(255, 255, 255, 0.6);
}
.top-nav span { cursor: pointer; transition: all 0.3s ease; }
.top-nav span:hover { color: #ffffff !important; text-shadow: 0 0 15px rgba(255, 255, 255, 0.6); }
.glass-panel {
    background: rgba(255, 255, 255, 0.03) !important; 
    backdrop-filter: blur(35px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(35px) saturate(200%) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.22) !important; 
    border-left: 1px solid rgba(255, 255, 255, 0.18) !important; 
    border-radius: 24px !important; padding: 35px !important;
    box-shadow: 0 30px 70px rgba(0, 0, 0, 0.5) !important;
}
.glass-panel .form, .glass-panel .fieldset, .glass-panel .padded,
.glass-panel div[class*="container"], .glass-panel div[class*="form"], .glass-panel div[class*="fieldset"] {
    background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important;
}
.glass-panel input, .glass-panel textarea, .glass-panel .gr-box {
    background: rgba(255, 255, 255, 0.06) !important; border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important; color: #ffffff !important; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.glass-panel input { padding: 14px 16px !important; margin-bottom: 14px !important; }
.glass-panel input:focus { background: rgba(255, 255, 255, 0.12) !important; border-color: rgba(255, 255, 255, 0.4) !important; }
.password-row-layout { display: flex !important; align-items: flex-end !important; gap: 12px !important; width: 100% !important; margin-bottom: 8px !important; }
.password-row-layout > div:first-child { flex-grow: 1 !important; }
.inline-toggle-btn {
    background: rgba(255, 255, 255, 0.06) !important; border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important; color: #ffffff !important; height: 52px !important; padding: 0 18px !important;
    font-weight: 600 !important; font-size: 13px !important; cursor: pointer !important; margin-bottom: 14px !important; 
}
.glass-panel button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important; border: none !important;
    color: white !important; font-weight: 600 !important; border-radius: 12px !important; padding: 15px !important;
    width: 100% !important; box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3) !important; transition: all 0.3s ease !important;
}
.status-box { margin-top: 20px; border-radius: 12px; text-align: center; }
.four-row-gallery div.gallery { grid-template-columns: repeat(5, minmax(0, 1fr)) !important; gap: 12px !important; }
"""

glass_theme = gr.themes.Soft().set(
    block_background_fill="rgba(255, 255, 255, 0.03)",
    input_background_fill="rgba(255, 255, 255, 0.03)",
    input_border_color="rgba(255, 255, 255, 0.1)"
)

# --- 7. UNIFIED GRADIO INTERFACE LAYOUT ---
with gr.Blocks(css=custom_css, theme=glass_theme) as demo:
    
    # HTML Dynamic Background Canvas Injection
    gr.HTML("""
    <canvas id="dynamic-bg-canvas"></canvas>
    <script>
        const canvas = document.getElementById('dynamic-bg-canvas');
        const ctx = canvas.getContext('2d');
        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;
        window.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });
        class NebulaOrb {
            constructor(x, y, radius, color, speedX, speedY) {
                this.x = x; this.y = y; this.radius = radius; this.color = color; this.speedX = speedX; this.speedY = speedY;
            }
            update() {
                this.x += this.speedX; this.y += this.speedY;
                if (this.x < -this.radius || this.x > width + this.radius) this.speedX *= -1;
                if (this.y < -this.radius || this.y > height + this.radius) this.speedY *= -1;
            }
            draw() {
                const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius);
                gradient.addColorStop(0, this.color); gradient.addColorStop(1, 'transparent');
                ctx.fillStyle = gradient; ctx.beginPath(); ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2); ctx.fill();
            }
        }
        const orbs = [
            new NebulaOrb(width * 0.2, height * 0.3, 450, 'rgba(236, 72, 153, 0.25)', 0.4, 0.3),
            new NebulaOrb(width * 0.8, height * 0.2, 500, 'rgba(56, 189, 248, 0.25)', -0.3, 0.5),
            new NebulaOrb(width * 0.5, height * 0.8, 550, 'rgba(139, 92, 246, 0.25)', 0.5, -0.4)
        ];
        function renderLoop() {
            ctx.fillStyle = '#0a0a12'; ctx.fillRect(0, 0, width, height);
            orbs.forEach(orb => { orb.update(); orb.draw(); });
            requestAnimationFrame(renderLoop);
        }
        renderLoop();
    </script>
    """)
    
    gr.HTML('<div class="top-nav"><span>PRODUCT</span><span>DEMO</span><span>PRICING</span></div>')

    # ================= PAGE 1: USER REGISTRATION / GATEWAY LAYER =================
    with gr.Column(visible=True) as auth_view:
        with gr.Row():
            gr.Markdown("""
            # 🚀 Welcome to Our Innovation Platform
            ### Experience the power of next-generation AI tools built just for you.
            """)
        gr.HTML("<br>")
        with gr.Row():
            # Left Side Gallery Preview
            with gr.Column(scale=1, elem_classes=["glass-panel"]):
                gr.Markdown("## Why Join Us?\n* **Instant Access:** Build, test, and deploy with top-tier AI pipelines.\n* **Cloud-Powered:** Heavy computational lift handled seamlessly.")
                gr.Markdown("### Features Preview")
                gr.Gallery(value=images_list, columns=5, rows=4, height="auto", object_fit="cover", show_label=False, container=False, interactive=False, elem_classes=["four-row-gallery"])
                
            # Right Side Inputs Form
            with gr.Column(scale=1, elem_classes=["glass-panel"]):
                gr.Markdown("## Create Your Free Account")
                reg_user = gr.Textbox(label="Username", placeholder="e.g., ai_creator99", max_lines=1)
                reg_email = gr.Textbox(label="Email Address", placeholder="you@example.com", max_lines=1)
                
                with gr.Row(elem_classes=["password-row-layout"]):
                    reg_pass = gr.Textbox(label="Password", placeholder="Choose a secure password", type="password", max_lines=1, container=False)
                    toggle_pass_btn = gr.Button("👁️ Show", elem_classes=["inline-toggle-btn"], min_width=75)
                    
                with gr.Row(elem_classes=["password-row-layout"]):
                    reg_confirm_pass = gr.Textbox(label="Confirm Password", placeholder="Repeat your password", type="password", max_lines=1, container=False)
                    toggle_confirm_btn = gr.Button("👁️ Show", elem_classes=["inline-toggle-btn"], min_width=75)
                
                register_btn = gr.Button("Register Now", variant="primary")
                status_output = gr.Markdown(elem_classes=["status-box"])

    # ================= PAGE 2: STUDIO GENERATION WORKBENCH (HIDDEN ON START) =================
    with gr.Column(visible=False) as studio_view:
        gr.Markdown("# 🎨 AI Image Generation Studio (Batch Mode)")
        
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["glass-panel"]):
                gr.Markdown("### 🐶 Live Inspo Stream 🐱")
                img_index = gr.State(value=0)
                slider_display = gr.Image(value=SLIDER_IMAGES[0], label="Slideshow", interactive=False, height=250)
                slider_timer = gr.Timer(value=3.0, active=True)
                
                slider_timer.tick(fn=rotate_slider, inputs=img_index, outputs=[slider_display, img_index], concurrency_limit=None)

        gr.HTML("<br>")
        with gr.Row():
            with gr.Column(elem_classes=["glass-panel"]) as control_col:
                mode = gr.Radio(choices=["Sketch to Image", "Text to Image"], value="Text to Image", label="1. Select Mode")
                prompt = gr.Textbox(value="A highly detailed charcoal drawing of a futuristic engine, steam and smoke", label="Prompt", lines=3)
                num_images = gr.Slider(minimum=1, maximum=100, value=1, step=1, label="Number of Images to Generate (Parallel Batch)")
                
                with gr.Group(visible=False) as sketch_inputs:
                    sketch_img = gr.Image(type="pil", label="Upload or Draw Sketch", sources=["upload", "clipboard"])
                    
                generate_btn = gr.Button("Generate Batch", variant="primary")

            with gr.Column(elem_classes=["glass-panel"]):
                processed_preview = gr.Image(label="Processed Edge Map Preview", type="pil", visible=False)
                output_gallery = gr.Gallery(label="Generated Output Images", columns=2, rows=None, object_fit="contain")

    # --- 8. EVENTS & TRANSITIONS ENGINE ---
    pass_visible = gr.State(False)
    confirm_visible = gr.State(False)

    def handle_toggle(current_state):
        new_state = not current_state
        return new_state, gr.update(type="text" if new_state else "password"), gr.update(value="🙈 Hide" if new_state else "👁️ Show")

    toggle_pass_btn.click(fn=handle_toggle, inputs=pass_visible, outputs=[pass_visible, reg_pass, toggle_pass_btn])
    toggle_confirm_btn.click(fn=handle_toggle, inputs=confirm_visible, outputs=[confirm_visible, reg_confirm_pass, toggle_confirm_btn])

    # Connect registration routing action
    register_btn.click(
        fn=save_user,
        inputs=[reg_user, reg_email, reg_pass, reg_confirm_pass],
        outputs=[status_output, auth_view, studio_view]
    )

    # Internal pipeline changes inside production block
    def update_ui(mode_selection):
        if mode_selection == "Text to Image":
            return gr.update(visible=False), gr.update(visible=False), gr.update(interactive=True)
        else:
            return gr.update(visible=True), gr.update(visible=True), gr.update(interactive=False)

    mode.change(fn=update_ui, inputs=mode, outputs=[sketch_inputs, processed_preview, prompt])
    
    generate_btn.click(
        fn=generate,
        inputs=[mode, sketch_img, prompt, num_images],
        outputs=[processed_preview, output_gallery]
    )

if __name__ == "__main__":
    init_db()
    demo.queue(default_concurrency_limit=4).launch(share=False)
