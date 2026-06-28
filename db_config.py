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
def save_user(username, email, password, repeat_password, month, day, year):
    """Validates parameters and saves user to MySQL database with optional DOB."""
    username = username.strip()
    email = email.strip()
    
    if not username or not email or not password or not repeat_password:
        return "⚠️ All fields are required!", False
        
    if password != repeat_password:
        return "❌ Passwords do not match. Please try again.", False
    
    # Check if a valid DOB dropdown configuration is selected; otherwise set to None (NULL)
    if month == "Select Month" or day == "Select Day" or year == "Select Year":
        dob_value = None
    else:
        dob_value = f"{year}-{month}-{day}"
        
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            
            # Check if username exists
            cursor.execute("SELECT Username FROM users WHERE Username = %s", (username,))
            if cursor.fetchone():
                return f"❌ Username '{username}' is already taken.", False
                
            # Check if email exists
            cursor.execute("SELECT Email FROM users WHERE Email = %s", (email,))
            if cursor.fetchone():
                return "❌ This email is already registered.", False
            
            secure_password_string = hash_password(password)
            insert_query = """
            INSERT INTO users (Username, Email, Password, dob) 
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (username, email, secure_password_string, dob_value))
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

# --- MODULAR GRADIO UI FUNCTION ---
def create_login_ui():
    """Generates the login screen components and assigns their internal events."""
    with gr.Group() as auth_container:
        with gr.Row():
            
            # COLUMN 1: Image Grid Gallery Module (Left Side)
            with gr.Column(scale=4):
                gr.Gallery(
                    value=images_list,
                    columns=4,          
                    rows=3,             
                    object_fit="cover", 
                    show_label=False,
                    container=False,     
                    interactive=False
                )
            
            # COLUMN 2: Authentication Tabs Area (Right Side)
            with gr.Column(scale=3):
                with gr.Tabs() as auth_tabs:
                    
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
                            
                            gr.Markdown("<label style='display:block; font-weight:700; font-size:14px; margin-top:14px; margin-bottom:8px;'>Please enter your date of birth (Optional):</label>")

                            with gr.Row():
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
                            '<div style="text-align: center; margin-top: 18px;">'
                            '    <a href="#" style="font-size: 13px; text-decoration: underline;">Forgot Password?</a>'
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

    return (
        login_user_input, login_pass, login_btn, login_status,
        reg_user, reg_email, reg_pass, reg_repeat_pass,
        birth_month, birth_day, birth_year,
        register_btn, register_status, reg_show_pass
    )

# --- APP MOUNT & ORCHESTRATION ---
with gr.Blocks() as demo:
    (
        login_user_input, login_pass, login_btn, login_status,
        reg_user, reg_email, reg_pass, reg_repeat_pass,
        birth_month, birth_day, birth_year,
        register_btn, register_status, reg_show_pass
    ) = create_login_ui()
    
    register_btn.click(
        fn=save_user,
        inputs=[reg_user, reg_email, reg_pass, reg_repeat_pass, birth_month, birth_day, birth_year],
        outputs=[register_status, gr.State()]
    )
    
    login_btn.click(
        fn=login_user,
        inputs=[login_user_input, login_pass],
        outputs=[login_status, gr.State()]
    )

if __name__ == "__main__":
    init_db()
    demo.launch(share=True)
