import os
import pandas as pd
import gradio as gr

#--- DATABASE CONFIGURATION USING PANDAS ---
DB_FILE = "users_db.csv"

def load_database():
    """Loads the user database or creates a new one if it doesn't exist."""
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    else:
        return pd.DataFrame(columns=["Username", "Email", "Password"])

def save_user(username, email, password, repeat_password, month, day, year):
    """Validates and saves a new user to the Pandas DataFrame."""
    df = load_database()
    
    username = username.strip()
    email = email.strip()
    
    if not username or not email or not password or not repeat_password:
        return "⚠️ All fields are required!", False
        
    if password != repeat_password:
        return "❌ Passwords do not match. Please try again." , False
    
    if username in df["Username"].values:
        return f"❌ Username '{username}' is already taken.", False
    
    if email in df["Email"].values:
        return "❌ This email is already registered."
    
    new_user = pd.DataFrame([{"Username": username, "Email": email, "Password": password}])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(DB_FILE, index=False)
    
    return f"🎉 Welcome aboard, {username}! Registration successful.", True

def login_user(username_or_email, password):
    """Checks credentials against the stored database file."""
    df = load_database()
    username_or_email = username_or_email.strip()
    
    if not username_or_email or not password:
        return "⚠️ Please fill in all fields!", False
        
    user_match = df[(df["Username"] == username_or_email) | (df["Email"] == username_or_email)]
    
    if user_match.empty:
        return "❌ Account not found. Please register first.", False
        
    if str(user_match.iloc[0]["Password"]) == str(password):
        return f"🔓 Welcome back, {user_match.iloc[0]['Username']}! Login successful." , True
    else:
        return "❌ Incorrect password. Please try again.", False

# --- STABLE & VERIFIED ASSETS GRID ---
images_list = [
    "https://picsum.photos/id/1015/400/300",
    "https://picsum.photos/id/1016/400/300",
    "https://picsum.photos/id/1018/400/300",
    "https://picsum.photos/id/1019/400/300",
    "https://picsum.photos/id/1022/400/300",
    "https://picsum.photos/id/1025/400/300",
    "https://picsum.photos/id/1035/400/300",
    "https://picsum.photos/id/1039/400/300",
    "https://picsum.photos/id/1043/400/300",
    "https://picsum.photos/id/1044/400/300",
    "https://picsum.photos/id/1045/400/300",
    "https://picsum.photos/id/1047/400/300"
]

# Global CSS context
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
/* Directly target the visual layout wrappers where forms and elements live */
.gradio-container div[class*="gallery"] {
    background: rgba(0, 0, 0, 0.35) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 20px !important;
    padding: 20px !important;
}

/* Target the deepest child form wrapper inside your custom glass column block */
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

/* Stop the inner group from inheriting box-shadow, boundaries, and background glass properties */
.gradio-container .unified-form-group,
.gradio-container div[class*="unified-form-group"] {
    background: transparent !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    width: 100% !important;
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

/* Unselected Tabs Default State */
#custom-auth-tabs div.tab-nav button:not(.selected) {
    color: rgba(255, 255, 255, 0.5) !important; 
    background: transparent !important;
    background-color: transparent !important;
}

/* ✨ ULTRA HIGH SPECIFICITY FIX: Wipes out Gradio's internal hover design tokens entirely ✨ */
#custom-auth-tabs div.tab-nav button:not(.selected):hover,
#custom-auth-tabs div.tab-nav button:not(.selected):hover span,
.gradio-container #custom-auth-tabs div.tab-nav button:not(.selected):hover,
.gradio-container #custom-auth-tabs div.tab-nav button:not(.selected):hover span {
    background: transparent !important;
    background-color: transparent !important;
    color: #ff9900 !important;
    --button-secondary-background-fill-hover: transparent !important;
    --button-secondary-text-color-hover: #ff9900 !important;
}

/* Selected Tab State */
#custom-auth-tabs div.tab-nav button.selected {
    color: #ffffff !important;
    background: rgba(236, 72, 153, 0.2) !important; 
    border-bottom: 3px solid #ec4899 !important; 
}

/* ============================================================
   FIX: Make inner tab hover text ORANGE (no white background)
============================================================ */
#custom-auth-tabs div.tab-nav button:not(.selected):hover,
#custom-auth-tabs div.tab-nav button:not(.selected):hover span {
    background: transparent !important;
    background-color: transparent !important;
    color: #ff9900 !important;        /* orange hover text */
    opacity: 1 !important;

    /* Kill ALL Gradio hover tokens */
    --button-secondary-background-fill-hover: transparent !important;
    --button-secondary-text-color-hover: #ff9900 !important;
    --button-primary-background-fill-hover: transparent !important;
    --button-primary-text-color-hover: #ff9900 !important;
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
   6. SUBMIT BUTTONS & INTERACTIVE ELEMENTS
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

.gradio-container button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(236, 72, 153, 0.5) !important;
}

/* Centering layout for forgot password layout link element */
.forgot-password-container {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
    margin-top: 18px !important;
}

.forgot-password-link {
    color: rgba(255, 255, 255, 0.6) !important;
    text-decoration: underline !important;
    font-size: 13px !important;
    transition: color 0.2s ease !important;
}

.forgot-password-link:hover {
    color: #ff9900 !important;
}

footer { display: none !important; }

/* ============================================================
   FINAL OVERRIDE: Force ORANGE hover text on both tabs
   (Create Account / Sign In) and kill white hover background
============================================================ */
#custom-auth-tabs div.tab-nav button:hover,
#custom-auth-tabs div.tab-nav button:hover *,
#custom-auth-tabs div.tab-nav button:not(.selected):hover,
#custom-auth-tabs div.tab-nav button:not(.selected):hover *,
#custom-auth-tabs div.tab-nav button:not(.selected):hover::before,
#custom-auth-tabs div.tab-nav button:not(.selected):hover::after {
    background: transparent !important;
    background-color: transparent !important;
    color: #ff9900 !important;
    opacity: 1 !important;

    /* Kill ALL Gradio hover tokens */
    --button-secondary-background-fill-hover: transparent !important;
    --button-secondary-text-color-hover: #ff9900 !important;
    --button-primary-background-fill-hover: transparent !important;
    --button-primary-text-color-hover: #ff9900 !important;

    /* Kill Material-style ripple/overlay */
    box-shadow: none !important;
}
/* ============================================================
   GLOBAL OVERRIDE: Fix tab hover color for Gradio theme
============================================================ */
:root {
    /* Kill white hover background */
    --button-secondary-background-fill-hover: transparent !important;
    --button-primary-background-fill-hover: transparent !important;

    /* Force orange hover text */
    --button-secondary-text-color-hover: #ff9900 !important;
    --button-primary-text-color-hover: #ff9900 !important;
}

/* Direct override for the actual tab buttons */
#custom-auth-tabs div.tab-nav button:hover,
#custom-auth-tabs div.tab-nav button:hover * {
    background: transparent !important;
    color: #ff9900 !important;
}

/* ============================================================
   RISE-UP HOVER EFFECT FOR AUTH TABS
============================================================ */
#custom-auth-tabs div.tab-nav button:not(.selected):hover {
    background: rgba(255, 255, 255, 0.08) !important; /* faint glass glow */
    transform: translateY(-3px) !important;           /* lift upward */
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45) !important;
    border-bottom: 2px solid #ff9900 !important;      /* orange underline */
    color: #ff9900 !important;                        /* optional: orange text */
    transition: all 0.25s ease !important;
}

/* Make the selected tab rise too (optional) */
#custom-auth-tabs div.tab-nav button.selected:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 20px rgba(236, 72, 153, 0.45) !important;
    border-bottom-color: #ff9900 !important;
}


"""

# --- MODULAR GRADIO UI FUNCTION ---
def create_login_ui():
    """Generates the login screen components and assigns their internal events."""
    with gr.Group() as auth_container:
        with gr.Row(elem_classes=["transparent-row"]):
            
            # COLUMN 1: Image Grid Gallery Module (Left Side)
            with gr.Column(scale=4, elem_classes=["gallery-container"]):
                gr.Gallery(
                    value=images_list,
                    columns=4,          
                    rows=3,             
                    object_fit="cover", 
                    show_label=False,
                    container=False,     
                    interactive=False
                )
            
            # COLUMN 2: High Contrast Authentication Tabs Area (Right Side)
            with gr.Column(scale=3, elem_classes=["dark-auth-panel"]):
                # Added elem_id="custom-auth-tabs" to cleanly target this module with high-priority CSS rules
                with gr.Tabs(elem_id="custom-auth-tabs") as auth_tabs:
                    
                    # --- TAB 1: Create Account ---
                    with gr.Tab("Create Account", id="create_tab"):
                        gr.Markdown("<h2>Create Your Free Account</h2>")
                        
                        with gr.Group(elem_classes=["unified-form-group"]):
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
                            
                            reg_show_pass = gr.Checkbox(label="Show Password", interactive=True)
                            
                            gr.Markdown("<label style='display:block; font-weight:700; color:#ffffff; font-size:14px; margin-top:14px; margin-bottom:8px;'>Please enter your date of birth:</label>")

                            with gr.Row(elem_id="dob-flex-row"):
                                birth_month = gr.Dropdown(
                                    choices=["Select Month"] + ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                                    value="Select Month", show_label=False
                                )
                                birth_day = gr.Dropdown(
                                    choices=["Select Day"] + [str(i) for i in range(1, 32)],
                                    value="Select Day", show_label=False
                                )
                                birth_year = gr.Dropdown(
                                    choices=["Select Year"] + [str(i) for i in range(1920, 2027)],
                                    value="Select Year", show_label=False
                                )
                                                                                    
                        register_btn = gr.Button("Register Now", variant="primary")
                        register_status = gr.Markdown()
                
                    # --- TAB 2: Sign In ---
                    with gr.Tab("Sign In", id="signin_tab"):
                        gr.Markdown("<h2>Access Your Account</h2>")
                        
                        login_user_input = gr.Textbox(label="Username or Email", placeholder="Enter your credentials", max_lines=1)
                        login_pass = gr.Textbox(
                            label="Password", placeholder="Enter your security password", 
                            type="password", max_lines=1, elem_id="login_password_field"
                        )
                        
                        login_show_pass = gr.Checkbox(label="Show Password", interactive=True)
                        login_btn = gr.Button("Sign In", variant="primary")
                        login_status = gr.Markdown()
                        
                        gr.HTML(
                                '<div class="forgot-password-container">'
                                '    <a href="#" class="forgot-password-link">Forgot Password?</a>'
                                '</div>'
                            )

        # --- JAVASCRIPT TOGGLES ---
        login_show_pass.change(
            fn=None, inputs=[login_show_pass], outputs=[],
            js="(checked) => { const f = document.querySelector('#login_password_field input'); if(f) f.type = checked ? 'text' : 'password'; }"
        )

        reg_show_pass.change(
            fn=None, inputs=[reg_show_pass], outputs=[],
            js="""
            (checked) => { 
                const p1 = document.querySelector('#reg_password_field input'); 
                const p2 = document.querySelector('#reg_repeat_password_field input');
                if(p1) p1.type = checked ? 'text' : 'password'; 
                if(p2) p2.type = checked ? 'text' : 'password'; 
            }
            """
        )

    # Return all the UI elements for external event handling       
    return (
        login_user_input, login_pass, login_btn, login_status,
        reg_user, reg_email, reg_pass, reg_repeat_pass,
        birth_month, birth_day, birth_year,
        register_btn, register_status, reg_show_pass
    )