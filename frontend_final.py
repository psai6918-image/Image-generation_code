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

def save_user(username, email, password, repeat_password, month, day, year):
    """Validates and saves a new user to the Pandas DataFrame."""
    df = load_database()
    
    username = username.strip()
    email = email.strip()
    
    if not username or not email or not password or not repeat_password:
        return "⚠️ All fields are required!"
        
    if password != repeat_password:
        return "❌ Passwords do not match. Please try again."
    
    if username in df["Username"].values:
        return f"❌ Username '{username}' is already taken."
    
    if email in df["Email"].values:
        return "❌ This email is already registered."
    
    new_user = pd.DataFrame([{"Username": username, "Email": email, "Password": password}])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(DB_FILE, index=False)
    
    return f"🎉 Welcome aboard, {username}! Registration successful."

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
        return f"🔓 Welcome back, {user_match.iloc[0]['Username']}! Login successful.", True
    else:
        return "❌ Incorrect password. Please try again.", False

def toggle_password_visibility(show_password):
    """Dynamically changes all password boxes between 'password' and 'text' types."""
    if show_password:
        return gr.update(type="text"), gr.update(type="text"), gr.update(type="text")
    return gr.update(type="password"), gr.update(type="password"), gr.update(type="password")

# --- STABLE & VERIFIED ASSETS GRID ---
images_list = [
    "https://picsum.photos/id/1015/400/300",  # Mountain Horizon
    "https://picsum.photos/id/1016/400/300",  # Canyon Wilderness
    "https://picsum.photos/id/1018/400/300",  # Mystic Woodland
    "https://picsum.photos/id/1019/400/300",  # Golden Coastline
    "https://picsum.photos/id/1022/400/300",  # Aurora Skies
    "https://picsum.photos/id/1025/400/300",  # Wilderness Camp
    "https://picsum.photos/id/1035/400/300",  # Cosmic Waterfall
    "https://picsum.photos/id/1039/400/300",  # Forest Path
    "https://picsum.photos/id/1043/400/300",  # Celestial Spires
    "https://picsum.photos/id/1044/400/300",  # Stone Citadel
    "https://picsum.photos/id/1045/400/300",  # Ethereal Bridge
    "https://picsum.photos/id/1047/400/300",  # Desert Lights
    "https://picsum.photos/id/1048/400/300",  # Mountain Fog
    "https://picsum.photos/id/1050/400/300",  # Highland Topography
    "https://picsum.photos/id/1051/400/300",  # Deep Valley Overlook
    "https://picsum.photos/id/1053/400/300",  # Ocean Spray Pier
    "https://picsum.photos/id/1057/400/300",  # Crimson Sea Sunset
    "https://picsum.photos/id/1062/400/300",  # Golden Hour Ridge
    "https://picsum.photos/id/1067/400/300",  # Winter Forest Line
    "https://picsum.photos/id/1069/400/300"   # Neon Twilight Valley
]

# --- ULTRALIGHT GLASSMORPHISM CUSTOM STYLE INJECTION ---
LOGIN_CSS = """
/* ============================================================
   TOP GLOBAL NAVIGATION BAR (OUTERMOST TABS)
============================================================ */

/* Isolates the very first, top-level Tab bar container on your page */
.gradio-container > div.tabs > div.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 0px !important;
    padding: 10px 20px 20px 0px !important;
    display: flex !important;
    gap: 30px !important;
}

/* Formats the Top Navigation tab links */
.gradio-container > div.tabs > div.tab-nav > button {
    font-size: 14px !important;
    font-weight: 600 !important;
    text-transform: none !important; /* Removes uppercase if you want standard text styling */
    letter-spacing: 1.5px !important;
    background: transparent !important;
    border: none !important;
    padding: 0px 0px 8px 0px !important; /* Bottom cushion only */
    border-radius: 0px !important;
    opacity: 0.8 !important;
    transition: all 0.3s ease !important;
}

/* UNSELECTED TOP NAV LINK STATE */
.gradio-container > div.tabs > div.tab-nav > button:not(.selected) {
    color: rgba(255, 119, 51, 0.6) !important; /* Faded orange */
}
.gradio-container > div.tabs > div.tab-nav > button:not(.selected):hover {
    color: #ffffff !important;
    opacity: 1.0 !important;
    background: transparent !important;
}

/* SELECTED ACTIVE TOP NAV LINK STATE (e.g., "Login" tab is active) */
.gradio-container > div.tabs > div.tab-nav > button.selected {
    color: rgb(255, 119, 51) !important; /* Vivid accent orange active state */
    border-bottom: 2px solid rgb(255, 119, 51) !important;
    opacity: 1.0 !important;
}


/* ============================================================
   INNER FORM AUTH TABS (SIGN IN / CREATE ACCOUNT)
============================================================ */

/* Safely targets nested tabs that are hidden deep inside your login layout columns */
div[class*="column"] div.tabs > div.tab-nav {
    background: rgba(0, 0, 0, 0.4) !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 6px 6px 0 6px !important;
    display: flex !important;
    gap: 8px !important;
    margin-bottom: 15px !important;
}

/* Formats the Inner Authentication panel tab buttons */
div[class*="column"] div.tabs > div.tab-nav > button {
    font-size: 14px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease-in-out !important;
}

/* UNSELECTED INNER AUTH TAB */
div[class*="column"] div.tabs > div.tab-nav > button:not(.selected) {
    color: rgba(255, 255, 255, 0.5) !important; 
    background: transparent !important;
}
div[class*="column"] div.tabs > div.tab-nav > button:not(.selected):hover {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #ffffff !important;
}

/* SELECTED ACTIVE INNER AUTH TAB */
div[class*="column"] d/* ============================================================
   1. CORE BACKGROUND DEEP LAYOUT SETUP (FIXED)
============================================================ */
html, body, .gradio-container, grad-app {
    background: 
        radial-gradient(circle at 15% 50%, rgba(236, 72, 153, 0.45), transparent 45%),
        radial-gradient(circle at 85% 30%, rgba(56, 189, 248, 0.45), transparent 45%),
        radial-gradient(circle at 50% 90%, rgba(139, 92, 246, 0.45), transparent 45%),
        linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #020617 100%) !important;
    background-attachment: fixed !important;
    min-height: 100vh !important;
}

.gradio-container {
    padding: 30px !important;
}

/* Fixes global row backgrounds so the background gradient shines through */
.gradio-container div[class*="row"], 
.gradio-container .gr-row {
    background: transparent !important;
    gap: 30px !important;   
}

/* ============================================================
   2. STYLING THE PANELS (DARK BLUR GLASS EFFECTS)
============================================================ */
/* Target only the actual content wrapper boxes, leaving the main canvas alone */
.gradio-container div[class*="column"] > div.form,
.gradio-container div[class*="column"] > div.block,
.gradio-container .gr-form,
.gradio-container .gr-box {
    background: rgba(255, 255, 255, 0.04) !important; 
    backdrop-filter: blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 50px 0 rgba(0, 0, 0, 0.5) !important;
    padding: 20px !important;
}

/* Clear default bright panels on the outer component wrappers */
.gradio-container div[class*="column"], 
.gradio-container div[class*="group"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Typography Overrides - Header Visibility Boost */
.gradio-container h2, 
.gradio-container div[class*="markdown"] h2,
.gradio-container .prose h2 {
    color: #ffffff !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    font-weight: 800 !important;          /* Increases boldness thickness */
    font-size: 28px !important;           /* Makes the header larger and pronounced */
    letter-spacing: -0.5px !important;    /* Modern tight layout alignment */
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5) !important; /* Drops a dark mask beneath text for contrast */
    margin-bottom: 15px !important;
}

/* Subtle accent line underneath the header */
.gradio-container h2::after {
    content: '';
    display: block;
    width: 50px;
    height: 3px;
    background: linear-gradient(90deg, #ec4899, #8b5cf6);
    margin-top: 8px;
    border-radius: 2px;
}

/* Typography Overrides - Subheader Visibility Boost */
.gradio-container h4, 
.gradio-container div[class*="markdown"] h4,
.gradio-container .prose h4 {
    color: rgba(255, 255, 255, 0.9) !important; /* Semi-transparent bright white */
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    font-weight: 600 !important;          /* Sharp, medium-bold weight */
    font-size: 15px !important;           /* Clear, readable subheader sizing */
    letter-spacing: 0.2px !important;     /* Slightly widened for modern contrast */
    text-shadow: 0 1px 6px rgba(0, 0, 0, 0.6) !important; /* Drop shadow layer to fight background blobs */
    margin-top: 15px !important;
    margin-bottom: 10px !important;
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
    color: #ffffff !important;
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
div[class*="column"] div.tabs > div.tab-nav {
    background: rgba(0, 0, 0, 0.4) !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 6px 6px 0 6px !important;
    display: flex !important;
    gap: 8px !important;
    margin-bottom: 20px !important;
}

div[class*="column"] div.tabs > div.tab-nav > button {
    font-size: 14px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease-in-out !important;
}

div[class*="column"] div.tabs > div.tab-nav > button:not(.selected) {
    color: rgba(255, 255, 255, 0.5) !important; 
    background: transparent !important;
}
div[class*="column"] div.tabs > div.tab-nav > button:not(.selected):hover {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #ffffff !important;
}

div[class*="column"] div.tabs > div.tab-nav > button.selected {
    color: #ffffff !important;
    background: rgba(236, 72, 153, 0.2) !important; 
    border-bottom: 3px solid #ec4899 !important; 
}

/* ============================================================
   5. FORM FIELDS & INTERACTIVE ELEMENT STYLES
============================================================ */
/* Universal Variables for Input Elements */
:root, .gradio-container {
    --block-background-fill: rgba(0, 0, 0, 0.4) !important;
    --block-border-color: rgba(255, 255, 255, 0.15) !important;
    --input-background-fill: rgba(0, 0, 0, 0.5) !important;
    --input-border-color: rgba(255, 255, 255, 0.15) !important;
    --body-text-color: #ffffff !important;
    --body-text-color-subdued: rgba(255, 255, 255, 0.6) !important;
    --button-secondary-background-fill: rgba(255, 119, 51, 0.9) !important;
    --button-secondary-text-color: #ffffff !important;
}

/* Input Fields Tuning */
input, textarea, .gradio-container input[type="text"], .gradio-container input[type="password"] {
    background: rgba(0, 0, 0, 0.6) !important; 
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
input::placeholder, textarea::placeholder {
    color: rgba(255, 255, 255, 0.4) !important;
}

/* Primary Submit Button Fixes */
.gradio-container button.primary, 
.gradio-container button[class*="primary"] {
    background: linear-gradient(90deg, #ec4899, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 14px 24px !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
}
.gradio-container button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(236, 72, 153, 0.5) !important;
}

/* Gallery Background Mask Fixes */
.gradio-gallery, div[class*="gallery"] {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}

/* Lower Section Checkbox Formattings */
.gradio-container label[class*="checkbox"] {
    color: #ffffff !important;
    font-weight: 500 !important;
}

/* Force the birthdate drop-down selectors to stay rigidly side-by-side */
.gradio-container .dob-row-container,
.gradio-container div[class*="dob-row-container"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important; /* Prevents wrapping down to a second line */
    gap: 12px !important;         /* Consistent gap between containers */
    width: 100% !important;
    background: transparent !important;
    margin-bottom: 15px !important;
}

/* Force each dropdown block to occupy equal, balanced width inside the row */
.dob-row-container > div {
    flex: 1 !important;
    min-width: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* 1. Force the main row wrapper to display child components horizontally */
.gradio-container div.dob-row-container,
.gradio-container div[class*="dob-row-container"],
.gradio-container [class*="dob-row-container"] > div[class*="form"] {
    display: flex !important;
    flex-direction: row !important;  
    flex-wrap: nowrap !important;     
    justify-content: space-between !important;
    align-items: stretch !important;
    width: 100% !important;
    gap: 12px !important;            /* Clean horizontal spacing between selectors */
    background: transparent !important;
}

/* 2. Strip standard block paddings and constraints from the outer wrapper layers */
.gradio-container div[class*="dob-row-container"] > div,
.gradio-container div[class*="dob-row-container"] div[class*="block"] {
    display: block !important;
    flex: 1 !important;             /* Allows each component element to scale out evenly */
    min-width: 0 !important;         /* Permits proper browser shrinking */
    max-width: none !important;      /* REMOVED: This stops the parent shell from collapsing */
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* 3. Force Gradio's inner input modules to display text clearly and at full width */
.gradio-container .dob-dropdown,
.gradio-container .dob-dropdown div[class*="select"],
.gradio-container .dob-dropdown .wrap-inner,
.gradio-container .dob-dropdown .secondary-wrap,
.gradio-container .dob-dropdown input[type="text"] {
    width: 100% !important;          
    min-height: 44px !important;     
    background: rgba(0, 0, 0, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    color: #ffffff !important;        /* Restores your clear white font color */
    font-weight: 700 !important;      /* Makes option text clearly visible */
    font-size: 14px !important;
    
    /* REMOVES CONSTRAINTS: This stops the inner font from clipping or hiding */
    padding: 0px 10px !important;     /* Horizontal breathing room */
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    overflow: visible !important;     /* Ensures text strings don't disappear */
}

/* Specific fix to make sure the selected string inside the box stays white */
.gradio-container .dob-dropdown span[class*="singleValue"],
.gradio-container .dob-dropdown div[class*="value"] {
    color: #ffffff !important;
    font-weight: 700 !important;
    display: inline-block !important;
}


/* Complete Footer Cleansing */
footer, .gradio-container footer, .embed-menu {
    display: none !important;
    height: 0px !important;
}

"""

# --- GRADIO INTERFACE DESIGN ---
## with gr.Blocks(css=LOGIN_CSS, theme=gr.themes.Soft()) as demo:
def create_login_ui():
    with gr.Group() as auth_container:
    
        # --- TOP NAVIGATION BAR ---
        # gr.HTML(
        #     """
        #     <div class="top-nav">
        #         <span>PRODUCT</span>
        #         <span>DEMO</span>
        #         <span>PRICING</span>
        #     </div>
        #     """
        # )
        
        # Split Layout Framework
        with gr.Row():
            
            # COLUMN 1: Image Grid Gallery Module (Left Side)
            with gr.Column(scale=1, elem_classes=["glass-panel"]):
                gr.Gallery(
                    value=images_list,
                    columns=4,          
                    rows=5,             
                    object_fit="cover", 
                    show_label=False,
                    container=True,     
                    interactive=False
                )
            
            # COLUMN 2: High Contrast Authentication Tabs Area (Right Side)
            with gr.Column(scale=1, elem_classes=["glass-panel", "dark-auth-panel"]):
                with gr.Tabs() as auth_tabs:
        
        # ============================================================
        # TAB 1: Create Account Registration Layout Panel
        # ============================================================
                    with gr.Tab("Create Account"):
                        with gr.Group():  # Unified dark-glass container for registration
                            gr.Markdown("<h2>Create Your Free Account</h2>")
                            
                            reg_user = gr.Textbox(label="Username", placeholder="e.g., ai_creator99", max_lines=1)
                            reg_email = gr.Textbox(label="Email Address", placeholder="you@example.com", max_lines=1)
                            
                            reg_pass = gr.Textbox(
                                label="Password", 
                                placeholder="Choose a secure password", 
                                type="password", 
                                max_lines=1,
                                elem_id="reg_password_field" # Kept for JS toggler
                            )
                            reg_repeat_pass = gr.Textbox(
                                label="Repeat Password", 
                                placeholder="Confirm your password", 
                                type="password", 
                                max_lines=1,
                                elem_id="reg_repeat_password_field" # Targeted via JS below
                            )
                            
                            reg_show_pass = gr.Checkbox(label="Show Password", interactive=True, elem_id="reg_checkbox_toggle")
                            
                            # --- Birth Date Dropdowns Section ---
                            gr.Markdown("<h4>Please enter your date of birth:</h4>")
                            with gr.Row(elem_classes=["dob-row-container"]):
                                birth_month = gr.Dropdown(
                                    choices=["Select Month", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                                    value="Select Month",
                                    show_label=False,
                                    elem_id="dob-month",
                                    elem_classes=["dob-dropdown"]
                                )
                                birth_day = gr.Dropdown(
                                    choices=["Select Day"] + [str(i) for i in range(1, 32)],
                                    value="Select Day",
                                    show_label=False,
                                    elem_id="dob-day",
                                    elem_classes=["dob-dropdown"]
                                )
                                birth_year = gr.Dropdown(
                                    choices=["Select Year"] + [str(i) for i in range(1920, 2027)],
                                    value="Select Year",
                                    show_label=False,
                                    elem_id="dob-year",
                                    elem_classes=["dob-dropdown"]
                                )
                            
                            # Action Buttons inside Tab 1 Group
                            register_btn = gr.Button("Register Now", variant="primary")
                            register_status = gr.Markdown()
                    
                    # ============================================================
                    # TAB 2: Sign In Layout Panel
                    # ============================================================
                    with gr.Tab("Sign In"):
                        with gr.Group():  # Unified dark-glass container for login
                            gr.Markdown("<h2>Access Your Account</h2>")
                            
                            login_user_input = gr.Textbox(
                                label="Username or Email", 
                                placeholder="Enter your credentials", 
                                max_lines=1
                            )
                                            
                            login_pass = gr.Textbox(
                                label="Password", 
                                placeholder="Enter your security password", 
                                type="password", 
                                max_lines=1,
                                elem_id="login_password_field" # Kept for JS toggler
                            )
                            
                            login_btn = gr.Button("Sign In", variant="primary")
                            login_status = gr.Markdown()
                            
                            login_show_pass = gr.Checkbox(label="Show Password", interactive=True, elem_id="login_checkbox_toggle")

            # ============================================================
            # FIXED JAVASCRIPT EVENT LISTENERS (Updated for repeat field!)
            # ============================================================
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
    return (
        login_user_input, login_pass, login_btn, login_status,   # 0 to 3
        reg_user, reg_email, reg_pass, reg_repeat_pass,         # 4 to 7
        birth_month, birth_day, birth_year,                      # 8 to 10
        register_btn, register_status,                           # 11 and 12
        reg_show_pass                                            # 13 (Maps to show_pass_checkbox)
    )



    # --- SIGN IN ACTION EVENT HANDLING ---
    login_btn.click(
        fn=login_user,
        inputs=[login_user_input, login_pass],
        outputs=login_status
    )

    # --- REGISTRATION ACTION EVENT HANDLING ---
    register_btn.click(
        fn=save_user,
        inputs=[reg_user, reg_email, reg_pass, reg_repeat_pass, birth_month, birth_day, birth_year],
        outputs=register_status
    )

# if __name__ == "__main__":
    #   demo.launch(share=False)