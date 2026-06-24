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
        return "⚠️ Please fill in all fields!"
        
    user_match = df[(df["Username"] == username_or_email) | (df["Email"] == username_or_email)]
    
    if user_match.empty:
        return "❌ Account not found. Please register first."
        
    if str(user_match.iloc[0]["Password"]) == str(password):
        return f"🔓 Welcome back, {user_match.iloc[0]['Username']}! Login successful."
    else:
        return "❌ Incorrect password. Please try again."

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
custom_css = """
/* Core Background Setup */
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

/* Header Text Rules */
.gradio-container h2 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Custom Navigation Bar Links */
.top-nav {
    display: flex;
    gap: 30px;
    padding: 10px 20px 20px 0px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1.5px;
}
.top-nav span {
    cursor: pointer;
    transition: all 0.3s ease;
    color: rgba(255, 255, 255, 0.8) !important;
}
.top-nav span:hover {
    color: #ffffff !important;
}

/* Glass & Dark Auth Container Elements */
.glass-panel {
    background: rgba(255, 255, 255, 0.03) !important; 
    backdrop-filter: blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 20px !important;
    padding: 25px !important;
    box-shadow: 0 20px 50px 0 rgba(0, 0, 0, 0.4) !important;
}

.dark-auth-panel {
    background: rgba(8, 10, 20, 0.9) !important;
    border-radius: 20px !important;
}

/* ============================================================
   FORCED RADICAL TEXT OVERRIDES (TAB READABILITY & CONTRAST)
============================================================ */

/* Targets Gradio internal variables directly to eradicate dull gray tabs */
.gradio-container, :root, .tabs {
    --button-large-text-color: #ffffff !important;
    --button-small-text-color: #ffffff !important;
    --body-text-color-subdued: #ffffff !important;
    --neutral-600: #ffffff !important; /* Forces fallback dull system texts to white */
    --block-label-text-color: rgba(255, 255, 255, 0.85) !important; /* Textbox labels */
}

/* Target Tab Nav bar wrapper structural components */
div.tabs > div.tab-nav {
    background: rgba(0, 0, 0, 0.8) !important;
    border-bottom: 3px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 8px 8px 0 8px !important;
    display: flex !important;
    gap: 12px !important;
}

/* Force ALL state buttons (Selected & Unselected) to completely shed dull look */
div.tabs > div.tab-nav > button {
    font-size: 15px !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    opacity: 1.0 !important;
    border: none !important;
    padding: 14px 28px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease-in-out !important;
}

/* UNSELECTED STATE (Fix for dull/invisible Create Account or Sign In) */
div.tabs > div.tab-nav > button:not(.selected) {
    color: rgba(255, 255, 255, 0.75) !important; 
    background: rgba(255, 255, 255, 0.05) !important;
}
div.tabs > div.tab-nav > button:not(.selected):hover {
    background: rgba(255, 255, 255, 0.15) !important;
    color: #ffffff !important;
}

/* SELECTED ACTIVE STATE */
div.tabs > div.tab-nav > button.selected {
    color: #ffffff !important;
    background: rgba(236, 72, 153, 0.3) !important; 
    border-bottom: 4px solid #ec4899 !important; 
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.5) !important;
}

/* ============================================================
   FIX FOR THE DARK RED ACCIDENT UTILITY CHECKBOX LABELS
  ============================================================ */
.pass-toggle-container label,
.pass-toggle-container span {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Form Fields Customization */
.glass-panel input, .glass-panel textarea, .glass-panel .container {
    background: rgba(0, 0, 0, 0.5) !important; 
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

/* Clear out inner gallery scroll constraints */
.gradio-gallery {
    background: rgba(0, 0, 0, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    max-height: 740px !important;
    overflow-y: auto !important;
}

/* Hide Date of Birth text elements neatly */
#dob-month label span, #dob-day label span, #dob-year label span,
.dob-dropdown label span, #dob-month label, #dob-day label, #dob-year label {
    display: none !important;
    height: 0px !important;
    padding: 0px !important;
    margin: 0px !important;
}

/* Complete footer extraction */
footer {
    display: none !important;
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
                
                # TAB 1: Sign In Layout Panel
                with gr.Tab("Sign In"):
                    gr.Markdown("<h2>Access Your Account</h2>")
                    login_user_input = gr.Textbox(label="Username or Email", placeholder="Enter your credentials", max_lines=1)
                    
                    with gr.Group():
                        login_pass = gr.Textbox(
                            label="Password", 
                            placeholder="Enter your security password", 
                            type="password", 
                            max_lines=1
                        )
                    
                    login_btn = gr.Button("Sign In", variant="secondary")
                    login_status = gr.Markdown()

                # TAB 2: Create Account Registration Layout Panel
                with gr.Tab("Create Account"):
                    gr.Markdown("<h2>Create Your Free Account</h2>")
                    reg_user = gr.Textbox(label="Username", placeholder="e.g., ai_creator99", max_lines=1)
                    reg_email = gr.Textbox(label="Email Address", placeholder="you@example.com", max_lines=1)
                                       
                    with gr.Group():
                        reg_pass = gr.Textbox(
                            label="Password", 
                            placeholder="Choose a secure password", 
                            type="password", 
                            max_lines=1
                        )
                        reg_repeat_pass = gr.Textbox(
                            label="Repeat Password", 
                            placeholder="Confirm your password", 
                            type="password", 
                            max_lines=1
                        )
                        
                    # --- Birth Date Dropdowns ---
                    with gr.Row():
                        birth_month = gr.Dropdown(
                            choices=["Select Month", "January", "February", "March", "April", "May", "June",
                                     "July", "August", "September", "October", "November", "December"],
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
                    
                    register_btn = gr.Button("Register Now", variant="primary")
                    register_status = gr.Markdown()
            
            # Encapsulated container ensuring checkbox labels are targetable for pure white color formatting
            with gr.Column(elem_classes=["pass-toggle-container"]):
                gr.HTML("<br>")
                show_pass_checkbox = gr.Checkbox(label="👁️ Show All Passwords Across Forms", value=False, container=False)

    # --- PASSWORD TOGGLE EVENT HANDLING ---
    show_pass_checkbox.select(
        fn=toggle_password_visibility,
        inputs=show_pass_checkbox,
        outputs=[login_pass, reg_pass, reg_repeat_pass]
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

if __name__ == "__main__":
    demo.launch(share=False)
