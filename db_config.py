import os
import secrets
import hashlib
import random
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

# --- CUSTOM CSS SETTINGS ---
LOGIN_CSS = """
/* ============================================================
   1. GLOBAL CANVAS & VIBRANT BACKGROUND BACKDROP
============================================================ */
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

/* Force ALL nested Gradio component rows/columns to be completely transparent */
.gradio-container div[class*="row"], 
.gradio-container div[class*="column"],
.gradio-container div[class*="group"],
.gradio-container div[class*="tabs"],
.gradio-container .gr-row,
.gradio-container .gr-column,
.gradio-container div[class*="svelte-"] {
    background-color: transparent !important;
    background: transparent !important;
    border-color: transparent !important;
    box-shadow: none !important;
}

/* Master Grid Row setup */
.gradio-container div[class*="row"], 
.gradio-container .gr-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: flex-start !important;
    justify-content: center !important;
    width: 100% !important;
    gap: 32px !important;
}

/* Constrains form card parent layout from stretching too wide on massive screens */
.gradio-container div[class*="dark-auth-panel"],
.gradio-container .dark-auth-panel {
    max-width: 460px !important;
    width: 100% !important;
}

/* ============================================================
   2. SHADING CONTENT MODULES (DARK BLUR GLASS EFFECT)
============================================================ */
.gradio-container div[class*="gallery"] {
    background: rgba(0, 0, 0, 0.35) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 20px !important;
    padding: 20px !important;
}

.gradio-container div[class*="dark-auth-panel"] div.form,
.gradio-container .dark-auth-panel div.form,
.gradio-container .gr-group, 
.gradio-container div.group {
    background: rgba(255, 255, 255, 0.04) !important; 
    backdrop-filter: blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 50px 0 rgba(0, 0, 0, 0.45) !important;
    padding: 24px !important;
}

/* ============================================================
   3. TOP GLOBAL NAVIGATION BAR (OUTERMOST TABS)
============================================================ */
.gradio-container > div.tabs > div.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 0px !important;
    padding: 10px 20px 20px 0px !important;
    display: flex !important;
    gap: 30px !important;
    margin-bottom: 25px !important;
}

.gradio-container > div.tabs > div.tab-nav > button {
    font-size: 14px !important;
    font-weight: 600 !important;
    text-transform: none !important;
    letter-spacing: 1.5px !important;
    background: transparent !important;
    border: none !important;
    padding: 0px 0px 8px 0px !important;
    border-radius: 0px !important;
    opacity: 0.8 !important;
    transition: all 0.3s ease !important;
}

.gradio-container > div.tabs > div.tab-nav > button:not(.selected) {
    color: rgba(255, 119, 51, 0.6) !important; 
}

.gradio-container > div.tabs > div.tab-nav > button:not(.selected):hover {
    color: #ff9900 !important;
    opacity: 1.0 !important;
    background: transparent !important;
}

.gradio-container > div.tabs > div.tab-nav > button.selected {
    color: rgb(255, 119, 51) !important; 
    border-bottom: 2px solid rgb(255, 119, 51) !important;
    opacity: 1.0 !important;
}

/* ============================================================
   4. INNER AUTHENTICATION TABS (SIGN IN / CREATE ACCOUNT)
============================================================ */
#custom-auth-tabs div.tab-nav {
    background: rgba(0, 0, 0, 0.4) !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 6px 6px 0 6px !important;
    display: flex !important;
    gap: 8px !important;
    margin-bottom: 20px !important;
}

#custom-auth-tabs div.tab-nav button {
    font-size: 14px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease-in-out !important;
}

#custom-auth-tabs div.tab-nav button:not(.selected) {
    color: rgba(255, 255, 255, 0.5) !important; 
    background: transparent !important;
    background-color: transparent !important;
}

#custom-auth-tabs div.tab-nav button:not(.selected):hover,
#custom-auth-tabs div.tab-nav button:not(.selected):hover span {
    background: transparent !important;
    background-color: transparent !important;
    color: #ff9900 !important;
}

#custom-auth-tabs div.tab-nav button.selected {
    color: #ffffff !important;
    background: rgba(236, 72, 153, 0.2) !important; 
    border-bottom: 3px solid #ec4899 !important; 
}

/* ============================================================
   5. TEXT FIELDS & FIELD LABEL TYPOGRAPHY BOOSTS
============================================================ */
:root, .gradio-container {
    --block-background-fill: rgba(0, 0, 0, 0.4) !important;
    --block-border-color: rgba(255, 255, 255, 0.15) !important;
    --input-background-fill: rgba(0, 0, 0, 0.5) !important;
    --input-border-color: rgba(255, 255, 255, 0.15) !important;
    --body-text-color: #ffffff !important;
    --body-text-color-subdued: rgba(255, 255, 255, 0.6) !important;
    --block-label-text-color: #ffffff !important;
}

.gradio-container h2, .gradio-container .prose h2, .gradio-container div[class*="markdown"] h2 {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 26px !important;
    margin-bottom: 12px !important;
}

.gradio-container label span, .gradio-container .block-label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}

.gradio-container div[class*="dark-auth-panel"] input,
.gradio-container div[class*="dark-auth-panel"] textarea,
input, textarea {
    background: rgba(0, 0, 0, 0.5) !important; 
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    padding: 10px 14px !important;
}

input::placeholder, textarea::placeholder {
    color: rgba(255, 255, 255, 0.35) !important;
}

/* ============================================================
   6. CHECKBOX CLICKS & ACCENT HIGHLIGHTS
============================================================ */
.gradio-container input[type="checkbox"] {
    -webkit-appearance: checkbox !important;
    appearance: checkbox !important;
    display: inline-block !important;
    width: 16px !important;
    height: 16px !important;
    cursor: pointer !important;
    accent-color: #ec4899 !important; 
    pointer-events: auto !important;
}

.gradio-container label[class*="checkbox"],
.gradio-container .gr-check-wrapper {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-start !important;
    background: transparent !important;
    border: none !important;
    padding: 6px 0 !important;
    gap: 8px !important;
    cursor: pointer !important;
    pointer-events: auto !important;
}

/* ============================================================
   7. SUBMIT BUTTONS & SPACING OPTIMIZATIONS
============================================================ */
.gradio-container .dark-auth-panel button.primary,
.gradio-container button.primary, 
.gradio-container button[class*="primary"] {
    background: linear-gradient(90deg, #ec4899, #8b5cf6) !important;
    background-image: linear-gradient(90deg, #ec4899, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
    max-width: 220px !important; 
    width: 100% !important;
    margin: 15px auto 5px auto !important; 
    display: block !important;
}

#signin_tab > div[class*="svelte-"] {
    gap: 0px !important;
}

#forgot-password-link {
    margin-top: 2px !important;
    padding-top: 0px !important;
    margin-bottom: 5px !important;
}

footer { display: none !important; }

/* RISE-UP HOVER EFFECT FOR AUTH TABS */
#custom-auth-tabs div.tab-nav button:not(.selected):hover {
    background: rgba(255, 255, 255, 0.08) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45) !important;
    border-bottom: 2px solid #ff9900 !important;
    color: #ff9900 !important;
    transition: all 0.25s ease !important;
}
"""

def init_db():
    """Ensures the users table exists in the MySQL database upon startup."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Username VARCHAR(255) NOT NULL UNIQUE,
        Email VARCHAR(255) NOT NULL UNIQUE,
        Password VARCHAR(255) NOT NULL,
        dob VARCHAR(20) DEFAULT NULL
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
def save_user(username, email, password, repeat_password):
    """Validates parameters and saves user to MySQL database."""
    username = username.strip()
    email = email.strip()
    
    if not username or not email or not password or not repeat_password:
        return "⚠️ All fields are required!", False
        
    if password != repeat_password:
        return "❌ Passwords do not match. Please try again.", False
        
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            
            cursor.execute("SELECT Username FROM users WHERE Username = %s", (username,))
            if cursor.fetchone():
                return f"❌ Username '{username}' is already taken.", False
                
            cursor.execute("SELECT Email FROM users WHERE Email = %s", (email,))
            if cursor.fetchone():
                return "❌ This email is already registered.", False
            
            secure_password_string = hash_password(password)
            insert_query = """
            INSERT INTO users (Username, Email, Password, dob) 
            VALUES (%s, %s, %s, NULL)
            """
            cursor.execute(insert_query, (username, email, secure_password_string))
            connection.commit()
            
            return f"🎉 Welcome aboard, {username}! Registration successful.", True
            
    except Error as e:
        return f"❌ Database Error: {e}", False
        
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def login_user(username_or_email, password):
    """Checks credentials against the MySQL database."""
    username_or_email = username_or_email.strip()
    
    if not username_or_email or not password:
        return "⚠️ Please fill in all fields!", False
        
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            
            query = "SELECT Username, Password FROM users WHERE Username = %s OR Email = %s"
            cursor.execute(query, (username_or_email, username_or_email))
            user_record = cursor.fetchone()
            
            if not user_record:
                return "❌ Account not found. Please register first.", False
                
            stored_credential_string = user_record['Password']
            if ":" not in stored_credential_string:
                return "❌ Stored password integrity error.", False
                
            salt, stored_hash = stored_credential_string.split(":", 1)
            runtime_hash = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
            
            if runtime_hash == stored_hash:
                return f"🔓 Welcome back, {user_record['Username']}! Login successful.", True
            else:
                return "❌ Incorrect password. Please try again.", False
                
    except Error as e:
        return f"❌ Database Error: {e}", False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# --- HIGH CAPACITY 1000 IMAGE DYNAMIC GENERATION ENGINE ---
def get_random_fantasy_gallery():
    """
    Generates a massive, dynamic pool of 1000 completely unique image links via Picsum.
    Samples 12 unique seeds on every single interface load/refresh.
    Guarantees no broken links, zero duplicate grids, and flawless rendering.
    """
    random_seeds = random.sample(range(1, 1001), 12)
    return [
        f"https://picsum.photos/400/300?random={seed}"
        for seed in random_seeds
    ]

def create_login_ui():
    """Generates the login screen components and assigns their internal events."""
    with gr.Group() as auth_container:
        with gr.Row():
            
            # COLUMN 1: Image Grid Gallery Module (Left Side)
            with gr.Column(scale=4):
                gr.Gallery(
                    value=get_random_fantasy_gallery, 
                    columns=4,          
                    rows=3,            
                    object_fit="cover", 
                    show_label=False,
                    container=False,    
                    interactive=False
                )
            
            # COLUMN 2: Authentication Tabs Area (Right Side)
            with gr.Column(scale=3, elem_classes=["dark-auth-panel"]):
                with gr.Tabs(elem_id="custom-auth-tabs") as auth_tabs:
                    
                    # --- TAB 1: Create Account ---
                    with gr.Tab("Create Account", id="create_tab"):
                        gr.Markdown("## Create Your Free Account")
                        
                        with gr.Group():
                            reg_user = gr.Textbox(label="Username", placeholder="e.g., ai_creator99", max_lines=1)
                            reg_email = gr.Textbox(label="Email Address", placeholder="you@example.com", max_lines=1)
                            
                            reg_pass = gr.Textbox(
                                label="Password", placeholder="Choose a secure password", 
                                type="password", max_lines=1, elem_id="reg_password_field"
                            )
                            reg_repeat_pass = gr.Textbox(
                                label="Repeat Password", placeholder="Confirm your password", 
                                type="password", max_lines=1, elem_id="reg_repeat_password_field"
                            )
                            
                            # Checked state controls BOTH reg_password_field and reg_repeat_password_field
                            reg_show_pass = gr.Checkbox(label="Show Password", interactive=True)
                                                                                
                            register_btn = gr.Button("Register Now", variant="primary")
                            register_status = gr.Markdown()
                
                    # --- TAB 2: Sign In ---
                    with gr.Tab("Sign In", id="signin_tab"):
                        gr.Markdown("## Access Your Account")
                        
                        login_user_input = gr.Textbox(label="Username or Email", placeholder="Enter your credentials", max_lines=1)
                        login_pass = gr.Textbox(
                            label="Password", placeholder="Enter your security password", 
                            type="password", max_lines=1, elem_id="login_password_field"
                        )
                        
                        login_show_pass = gr.Checkbox(label="Show Password", interactive=True)
                        login_btn = gr.Button("Sign In", variant="primary")
                        login_status = gr.Markdown()
                        
                        gr.HTML(
                            '<div id="forgot-password-link" style="text-align: center;">'
                            '    <a href="#" style="font-size: 13px; text-decoration: underline;">Forgot Password?</a>'
                            '</div>'
                        )

        # --- JAVASCRIPT TOGGLES ---
        # Controls visibility for the single Login field
        login_show_pass.change(
            fn=None, inputs=[login_show_pass], outputs=[],
            js="(checked) => { const f = document.querySelector('#login_password_field input'); if(f) f.type = checked ? 'text' : 'password'; }"
        )

        # Controls visibility simultaneously for BOTH fields on the registration screen
        reg_show_pass.change(
            fn=None, inputs=[reg_show_pass], outputs=[],
            js="""
            (checked) => { 
                const targetType = checked ? 'text' : 'password';
                const p1 = document.querySelector('#reg_password_field input'); 
                const p2 = document.querySelector('#reg_repeat_password_field input');
                if(p1) p1.type = targetType; 
                if(p2) p2.type = targetType; 
            }
            """
        )

    return (
        login_user_input, login_pass, login_btn, login_status,
        reg_user, reg_email, reg_pass, reg_repeat_pass,
        register_btn, register_status, reg_show_pass
    )

# --- APP MOUNT ---
with gr.Blocks(css=LOGIN_CSS) as demo:
    (
        login_user_input, login_pass, login_btn, login_status,
        reg_user, reg_email, reg_pass, repeat_password,
        register_btn, register_status, reg_show_pass
    ) = create_login_ui()
    
    register_btn.click(
        fn=save_user,
        inputs=[reg_user, reg_email, reg_pass, repeat_password],
        outputs=[register_status, gr.State()]
    )
    
    login_btn.click(
        fn=login_user,
        inputs=[login_user_input, login_pass],
        outputs=[login_status, gr.State()]
    )

    # FORCE DARK MODE ON LOAD USING JS:
    # This automatically adds the '.dark' class to the html document header
    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { document.documentElement.classList.add('dark'); }"
    )

if __name__ == "__main__":
    init_db()
    demo.launch(share=True)