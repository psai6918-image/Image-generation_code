# pages/login.py
import os
import secrets
import hashlib
import random
import mysql.connector
from mysql.connector import Error
import gradio as gr

# Reference database configuration safely from separate settings config file
from config.database import DB_CONFIG

# --- DYNAMIC CSS LOADING ---
# This reads the external CSS file into the LOGIN_CSS variable so app.py doesn't break
css_path = os.path.join("assets", "login.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        LOGIN_CSS = f.read()
else:
    LOGIN_CSS = ""

# --- SECURITY & HASHING FUNCTIONS ---
def hash_password(password: str) -> str:
    """Generates a secure 16-byte random salt and hashes the password using SHA-256."""
    salt = secrets.token_hex(16)
    hashed_bytes = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    return f"{salt}:{hashed_bytes}"


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
            INSERT INTO users (Username, Email, Password) 
            VALUES (%s, %s, %s)
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


# --- PASSWORD RECOVERY ENGINES ---
def direct_forgot_password_trigger(username_or_email):
    """
    Directly triggered when user clicks 'Forgot Password?'. 
    Reads the value from the existing sign-in screen username field.
    """
    target = username_or_email.strip()
    if not target:
        return (
            "⚠️ Please enter your Username or Email address in the text box above before clicking Forgot Password.", 
            gr.update(), 
            gr.update(), 
            gr.update(), gr.update(), gr.update(), gr.update(value=False)
        )
        
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            query = "SELECT Username FROM users WHERE Username = %s OR Email = %s"
            cursor.execute(query, (target, target))
            user = cursor.fetchone()
            
            if user:
                matched_username = user[0]
                return (
                    f"✅ Account found for '{matched_username}'. Please choose your new password below.",
                    gr.update(visible=True),               
                    gr.update(selected="forgot_tab"),       
                    gr.update(visible=True),                
                    gr.update(visible=True),                
                    gr.update(visible=True),                
                    gr.update(visible=True, value=False)    
                )
            else:
                return (
                    "❌ No profile found matching that Username or Email.", 
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                )
    except Error as e:
        return f"❌ Database operational fault: {e}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


def commit_new_password(username_or_email, new_pass, repeat_pass):
    """Commits the new cryptographic string into the MySQL table data schema."""
    target = username_or_email.strip()
    new_pass = new_pass.strip()
    repeat_pass = repeat_pass.strip()
    
    if not new_pass or not repeat_pass:
        return "⚠️ Password fields cannot be empty."
    if new_pass != repeat_pass:
        return "❌ Passwords do not match. Please review."
        
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            
            secure_password_string = hash_password(new_pass)
            
            update_query = "UPDATE users SET Password = %s WHERE Username = %s OR Email = %s"
            cursor.execute(update_query, (secure_password_string, target, target))
            connection.commit()
            
            return "🎉 Password updated successfully! You can now switch back to the Sign In tab."
    except Error as e:
        return f"❌ Failed writing update to database: {e}"
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


# --- HIGH CAPACITY 1000 IMAGE DYNAMIC GENERATION ENGINE ---
def get_random_fantasy_gallery():
    random_seeds = random.sample(range(1, 1001), 12)
    return [f"https://picsum.photos/400/300?random={seed}" for seed in random_seeds]


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
                        
                        forgot_link_btn = gr.Button("Forgot Password?", elem_classes=["forgot-password-btn"])

                    # --- TAB 3: Forgot Password ---
                    with gr.Tab("Password Reset", id="forgot_tab", visible=False) as forgot_tab:
                        gr.Markdown("## Recover Account")
                        
                        reset_new_password = gr.Textbox(label="New Password", placeholder="Enter a new password", type="password", max_lines=1, visible=False, elem_id="reset_password_field")
                        reset_repeat_password = gr.Textbox(label="Confirm New Password", placeholder="Repeat your new password", type="password", max_lines=1, visible=False, elem_id="reset_repeat_password_field")
                        
                        reset_show_pass = gr.Checkbox(label="Show Password", interactive=True, visible=False)
                        reset_save_btn = gr.Button("Update Password", variant="primary", visible=False)
                        
                        forgot_status = gr.Markdown()
                        back_to_signin_btn = gr.Button("← Back to Sign In", elem_classes=["forgot-password-btn"])

        # --- JAVASCRIPT TOGGLES ---
        login_show_pass.change(
            fn=None, inputs=[login_show_pass], outputs=[],
            js="(checked) => { const f = document.querySelector('#login_password_field input'); if(f) f.type = checked ? 'text' : 'password'; }"
        )

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

        reset_show_pass.change(
            fn=None, inputs=[reset_show_pass], outputs=[],
            js="""
            (checked) => { 
                const targetType = checked ? 'text' : 'password';
                const p1 = document.querySelector('#reset_password_field input'); 
                const p2 = document.querySelector('#reset_repeat_password_field input');
                if(p1) p1.type = targetType; 
                if(p2) p2.type = targetType; 
            }
            """
        )

    return (
        login_user_input, login_pass, login_btn, login_status,
        reg_user, reg_email, reg_pass, reg_repeat_pass,
        register_btn, register_status, reg_show_pass,
        forgot_link_btn, forgot_tab, auth_tabs, forgot_status, back_to_signin_btn,
        reset_new_password, reset_repeat_password, reset_show_pass, reset_save_btn
    )