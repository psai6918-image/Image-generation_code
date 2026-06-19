import os
import pandas as pd
import gradio as gr

# --- DATABASE CONFIGURATION USING PANDAS ---
DB_FILE = "users_db.csv"

def load_database():
    """Loads the user database or creates a new one if it doesn't exist."""
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    else:
        return pd.DataFrame(columns=["Username", "Email", "Password"])

def save_user(username, email, password):
    """Validates and saves a new user to the Pandas DataFrame."""
    df = load_database()
    
    username = username.strip()
    email = email.strip()
    
    if not username or not email or not password:
        return "⚠️ All fields are required!"
    
    if username in df["Username"].values:
        return f"❌ Username '{username}' is already taken."
    
    if email in df["Email"].values:
        return "❌ This email is already registered."
    
    new_user = pd.DataFrame([{"Username": username, "Email": email, "Password": password}])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(DB_FILE, index=False)
    
    return f"🎉 Welcome aboard, {username}! Registration successful."

def toggle_password_visibility(show_password):
    """Dynamically changes the password text box type between 'password' and 'text'."""
    if show_password:
        return gr.update(type="text")
    return gr.update(type="password")

# --- IMAGE SOURCES ---
images_list = [
    "https://picsum.photos/300/300?random=1",
    "https://picsum.photos/300/300?random=2",
    "https://picsum.photos/300/300?random=3",
    "https://picsum.photos/300/300?random=4"
]

# --- CUSTOM CSS FOR ULTRA-GLASSMORPHISM STYLE ---
custom_css = """
/* Vibrant multi-layered background so the glass has colors to distort */
.gradio-container {
    background: 
        radial-gradient(circle at 15% 50%, rgba(236, 72, 153, 0.45), transparent 40%),
        radial-gradient(circle at 85% 30%, rgba(56, 189, 248, 0.45), transparent 40%),
        radial-gradient(circle at 50% 90%, rgba(139, 92, 246, 0.45), transparent 40%),
        linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #020617 100%) !important;
    background-attachment: fixed !important;
    min-height: 100vh;
    padding: 30px !important;
}

/* --- GLOBAL TEXT COLOR FIX --- */
.gradio-container h1, 
.gradio-container h2, 
.gradio-container h3, 
.gradio-container p,
.gradio-container span,
.gradio-container label,
.prose {
    color: #ffffff !important;
}

/* Custom Navigation Bar */
.top-nav {
    display: flex;
    gap: 30px;
    padding: 10px 20px 20px 0px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: rgba(255, 255, 255, 0.8);
}

.top-nav span {
    cursor: pointer;
    transition: all 0.3s ease;
    text-shadow: 0 0 10px rgba(255,255,255,0);
}

.top-nav span:hover {
    color: #ffffff !important;
    text-shadow: 0 0 10px rgba(255,255,255,0.5);
}

/* --- EXTREME GLASS PANELS --- */
.glass-panel {
    /* Lowered opacity for more transparency */
    background: rgba(255, 255, 255, 0.03) !important; 
    
    /* Pushed the blur and saturation much higher */
    backdrop-filter: blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
    
    /* Brighter, slightly thicker border to mimic glass edge */
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.3) !important; /* Top highlight */
    border-left: 1px solid rgba(255, 255, 255, 0.2) !important; /* Left highlight */
    border-radius: 20px !important;
    padding: 30px !important;
    
    /* Deep outer shadow + Inner highlight for depth */
    box-shadow: 
        0 20px 50px 0 rgba(0, 0, 0, 0.4), 
        inset 0 0 20px rgba(255, 255, 255, 0.05) !important;
}

/* Glassy inputs */
.glass-panel input, .glass-panel textarea {
    background: rgba(0, 0, 0, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.3) !important; /* Bottom highlight */
    border-radius: 8px !important;
    color: white !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3) !important;
    transition: all 0.3s ease;
}

.glass-panel input:focus, .glass-panel textarea:focus {
    background: rgba(0, 0, 0, 0.4) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
    outline: none !important;
}

/* Placeholder text color */
.glass-panel input::placeholder {
    color: rgba(255, 255, 255, 0.4) !important;
}
"""

# --- GRADIO INTERFACE DESIGN ---
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # --- TOP NAVIGATION BAR ---
    gr.HTML(
        """
        <div class="top-nav">
            <span>PRODUCT</span>
            <span>DEMO</span>
            <span>PRICING</span>
        </div>
        """
    )
    
    # Header / Welcome Section
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                # 🚀 Welcome to Our Innovation Platform
                ### Experience the power of next-generation AI tools built just for you.
                Discover seamless workflows, fast generation pipelines, and state-of-the-art models all in one workspace.
                """
            )
    
    gr.HTML("<br>") 
    
    # Split Layout: Left side info + Images, Right side form
    with gr.Row():
        # Left Column (Content & Square Images wrapped in Glass class)
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
                columns=2,          
                rows=2,             
                height=300,         
                object_fit="cover", 
                show_label=False,
                container=False,
                interactive=False
            )
            
        # Right Column (Registration Form wrapped in Glass class)
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.Markdown("## Create Your Free Account")
            
            reg_user = gr.Textbox(label="Username", placeholder="e.g., ai_creator99", max_lines=1)
            reg_email = gr.Textbox(label="Email Address", placeholder="you@example.com", max_lines=1)
            
            # Password block with view toggle option
            with gr.Group():
                reg_pass = gr.Textbox(
                    label="Password", 
                    placeholder="Choose a secure password", 
                    type="password", 
                    max_lines=1
                )
                show_pass_checkbox = gr.Checkbox(label="👁️ Show Password", value=False, container=False)
            
            register_btn = gr.Button("Register Now", variant="primary")
            status_output = gr.Markdown()

    # --- PASSWORD TOGGLE EVENT ---
    show_pass_checkbox.select(
        fn=toggle_password_visibility,
        inputs=show_pass_checkbox,
        outputs=reg_pass
    )

    # --- BUTTON CLICK EVENT ---
    register_btn.click(
        fn=save_user,
        inputs=[reg_user, reg_email, reg_pass],
        outputs=status_output
    )

if __name__ == "__main__":
    demo.launch(share=False)
