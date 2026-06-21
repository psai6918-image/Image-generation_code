import os
import secrets
import hashlib
import mysql.connector
from mysql.connector import Error
import gradio as gr

# --- DATABASE CONFIGURATION ---
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

# --- SECURITY & HASHING FUNCTIONS ---
def hash_password(password: str) -> str:
    """Generates a secure 16-byte random salt and hashes the password using SHA-256."""
    salt = secrets.token_hex(16)
    hashed_bytes = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    return f"{salt}:{hashed_bytes}"

# --- GRADIO ACTIONS ---
def save_user(username, email, password, confirm_password):
    """Validates parameters and saves user to MySQL database."""
    username = username.strip()
    email = email.strip()
    
    if not username or not email or not password or not confirm_password:
        return "### ⚠️ All fields are required!"
        
    if password != confirm_password:
        return "### ❌ Passwords do not match\nPlease verify both fields."
    
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            
            cursor.execute("SELECT Username FROM users WHERE Username = %s", (username,))
            if cursor.fetchone():
                return f"### ❌ Username '{username}' is already taken."
                
            cursor.execute("SELECT Email FROM users WHERE Email = %s", (email,))
            if cursor.fetchone():
                return "### ❌ This email is already registered."
            
            secure_password_string = hash_password(password)
            insert_query = """
            INSERT INTO users (Username, Email, Password) 
            VALUES (%s, %s, %s)
            """
            cursor.execute(insert_query, (username, email, secure_password_string))
            connection.commit()
            
            return f"### 🎉 Welcome aboard, {username}!\nRegistration successful."
            
    except Error as e:
        return f"### ❌ Database Error\nCould not process request: {e}"
        
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# --- IMAGE SOURCES (20 UNIQUE SQUARE IMAGES) ---
images_list = [
    f"https://picsum.photos/300/300?random={i}" for i in range(1, 21)
]

# --- PREMIUM LUXURY GLASSMORPHISM CSS ---
custom_css = """
.gradio-container {
    background: #000000 !important;
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
    background: #000000;
}
.gradio-container h1, 
.gradio-container h2, 
.gradio-container h3, 
.gradio-container p,
.gradio-container span,
.gradio-container label,
.gradio-container .prose {
    color: #ffffff !important;
}
.top-nav {
    display: flex;
    gap: 40px;
    padding: 10px 0 30px 0;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    color: rgba(255, 255, 255, 0.6);
}
.top-nav span {
    cursor: pointer;
    transition: all 0.3s ease;
}
.top-nav span:hover {
    color: #ffffff !important;
    text-shadow: 0 0 15px rgba(255, 255, 255, 0.6);
}
.glass-panel {
    background: rgba(255, 255, 255, 0.02) !important; 
    backdrop-filter: blur(35px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(35px) saturate(200%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.15) !important; 
    border-left: 1px solid rgba(255, 255, 255, 0.12) !important; 
    border-radius: 24px !important;
    padding: 35px !important;
    box-shadow: 0 30px 70px rgba(0, 0, 0, 0.9) !important;
}
.glass-panel .form, 
.glass-panel .fieldset, 
.glass-panel .padded,
.glass-panel div[class*="container"],
.glass-panel div[class*="form"],
.glass-panel div[class*="fieldset"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.glass-panel input, 
.glass-panel textarea,
.glass-panel .gr-box {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #ffffff !important; 
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.5) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.glass-panel input {
    padding: 14px 16px !important;
    margin-bottom: 14px !important;
}
.glass-panel input:focus {
    background: rgba(255, 255, 255, 0.10) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.1) !important;
}
.glass-panel label span {
    background: transparent !important;
    color: rgba(255, 255, 255, 0.9) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    margin-bottom: 8px !important;
    display: block !important;
    letter-spacing: 0.5px;
}
.password-row-layout {
    display: flex !important;
    align-items: flex-end !important; 
    gap: 12px !important;
    width: 100% !important;
    margin-bottom: 8px !important;
}
.password-row-layout > div:first-child {
    flex-grow: 1 !important;
}
.inline-toggle-btn {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    height: 52px !important; 
    padding: 0 18px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-bottom: 14px !important; 
}
.inline-toggle-btn:hover {
    background: rgba(255, 255, 255, 0.16) !important;
    border-color: rgba(255, 255, 255, 0.35) !important;
}
.glass-panel button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 15px !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3) !important;
    transition: all 0.3s ease !important;
    margin-top: 15px;
    letter-spacing: 0.5px;
}
.glass-panel button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5) !important;
}
.status-box {
    margin-top: 20px;
    border-radius: 12px;
    text-align: center;
}
.four-row-gallery div.gallery {
    grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
    gap: 12px !important;
}
"""

glass_theme = gr.themes.Soft().set(
    block_background_fill="rgba(0, 0, 0, 1)",
    input_background_fill="rgba(255, 255, 255, 0.03)",
    input_border_color="rgba(255, 255, 255, 0.1)"
)

# --- GRADIO INTERFACE DESIGN ---
with gr.Blocks(css=custom_css, theme=glass_theme) as demo:
    
    # Background Canvas: Render loop adjusted to purely map to clear canvas on #000000 
    gr.HTML(
        """
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
                    this.x = x;
                    this.y = y;
                    this.radius = radius;
                    this.color = color;
                    this.speedX = speedX;
                    this.speedY = speedY;
                }
                update() {
                    this.x += this.speedX;
                    this.y += this.speedY;
                    if (this.x < -this.radius || this.x > width + this.radius) this.speedX *= -1;
                    if (this.y < -this.radius || this.y > height + this.radius) this.speedY *= -1;
                }
                draw() {
                    const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius);
                    gradient.addColorStop(0, this.color);
                    gradient.addColorStop(1, 'transparent');
                    ctx.fillStyle = gradient;
                    ctx.beginPath();
                    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
            
            // Replaced the bright neon colors with dark, deep variants to respect the absolute black look
            const orbs = [
                new NebulaOrb(width * 0.2, height * 0.3, 450, 'rgba(120, 20, 60, 0.15)', 0.3, 0.2),
                new NebulaOrb(width * 0.8, height * 0.2, 500, 'rgba(20, 60, 120, 0.15)', -0.2, 0.3),
                new NebulaOrb(width * 0.5, height * 0.8, 550, 'rgba(60, 20, 120, 0.15)', 0.3, -0.2)
            ];
            
            function renderLoop() {
                ctx.fillStyle = '#000000'; 
                ctx.fillRect(0, 0, width, height);
                orbs.forEach(orb => {
                    orb.update();
                    orb.draw();
                });
                requestAnimationFrame(renderLoop);
            }
            renderLoop();
        </script>
        """
    )
    
    gr.HTML(
        """
        <div class="top-nav">
            <span>PRODUCT</span>
            <span>DEMO</span>
            <span>PRICING</span>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                # 🚀 Welcome to Our Innovation Platform
                ### Experience the power of next-generation AI tools built just for you.
                """
            )
    
    gr.HTML("<br>") 
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.Markdown(
                """
                ## Why Join Us?
                * **Instant Access:** Build, test, and deploy with top-tier AI pipelines.
                * **Cloud-Powered:** Heavy computational lift handled seamlessly.
                """
            )
            gr.Markdown("### Features Preview")
            gr.Gallery(
                value=images_list,
                columns=5, rows=4, height="auto", object_fit="cover", 
                show_label=False, container=False, interactive=False,
                elem_classes=["four-row-gallery"]
            )
            
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.Markdown("## Create Your Free Account")
            
            reg_user = gr.Textbox(label="Username", placeholder="e.g., ai_creator99", max_lines=1)
            reg_email = gr.Textbox(label="Email Address", placeholder="you@example.com", max_lines=1)
            
            with gr.Row(elem_classes=["password-row-layout"]):
                reg_pass = gr.Textbox(
                    label="Password", placeholder="Choose a secure password", 
                    type="password", max_lines=1, container=False
                )
                toggle_pass_btn = gr.Button("👁️ Show", elem_classes=["inline-toggle-btn"], min_width=75)
            
            with gr.Row(elem_classes=["password-row-layout"]):
                reg_confirm_pass = gr.Textbox(
                    label="Confirm Password", placeholder="Repeat your password", 
                    type="password", max_lines=1, container=False
                )
                toggle_confirm_btn = gr.Button("👁️ Show", elem_classes=["inline-toggle-btn"], min_width=75)
            
            register_btn = gr.Button("Register Now", variant="primary")
            
            status_output = gr.Markdown(elem_classes=["status-box"])

    # --- ACTION LISTENERS ---
    pass_visible = gr.State(False)
    confirm_visible = gr.State(False)

    def handle_toggle(current_state):
        new_state = not current_state
        return (
            new_state, 
            gr.Textbox(type="text" if new_state else "password"), 
            gr.Button(value="🙈 Hide" if new_state else "👁️ Show")
        )

    toggle_pass_btn.click(fn=handle_toggle, inputs=pass_visible, outputs=[pass_visible, reg_pass, toggle_pass_btn])
    toggle_confirm_btn.click(fn=handle_toggle, inputs=confirm_visible, outputs=[confirm_visible, reg_confirm_pass, toggle_confirm_btn])

    register_btn.click(
        fn=save_user,
        inputs=[reg_user, reg_email, reg_pass, reg_confirm_pass],
        outputs=status_output
    )

if __name__ == "__main__":
    init_db()
    demo.launch(share=True)