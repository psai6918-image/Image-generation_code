# app.py
import gradio as gr

# 1. MODULAR PAGE IMPORTS
from pages.login import (
    create_login_ui, LOGIN_CSS, save_user, login_user,
    direct_forgot_password_trigger, commit_new_password
)
from pages.dashboard import (
    create_generator_ui, GENERATOR_CSS, generate, on_gallery_select, 
    modify_selected_image, update_ui, set_processing_notice, 
    append_to_favorites, reset_to_original_image, load_existing_favorites
)
from pages.payment import create_payment_ui, PAYMENT_CSS
from pages.contact_us import create_contact_us_ui, CONTACT_US_CSS, handle_contact_submit

# 2. CORE DATABASE INITIALIZATION
from config.database import init_db
init_db()

# 3. CASCADING PRESENTATION LAYERS STYLES
COMBINED_CSS = LOGIN_CSS + "\n" + GENERATOR_CSS + "\n" + PAYMENT_CSS + "\n" + CONTACT_US_CSS

with gr.Blocks(css=COMBINED_CSS, title="AI Studio Workspace") as demo:
    
    # Global multi-user session state engine tracking the active username
    session_user = gr.State("anonymous")
    
    # ==========================================
    # VIEWPORT CONTAINERS (PUBLIC & PRIVATE LAYOUTS)
    # ==========================================
    
    # --- VIEW 1: AUTHENTICATION LAYOUT (PUBLIC) ---
    with gr.Column(visible=True) as public_layout:
        auth_elements = create_login_ui()
        
        # EXACT MATCH UNPACKING: Elements 0-19 map perfectly to underlying components
        login_user_input      = auth_elements[0]
        login_pass            = auth_elements[1]
        login_btn             = auth_elements[2]
        login_status          = auth_elements[3]
        
        reg_user              = auth_elements[4]
        reg_email             = auth_elements[5]
        reg_pass              = auth_elements[6]
        reg_repeat_pass       = auth_elements[7]
        register_btn          = auth_elements[8]
        register_status       = auth_elements[9]
        reg_show_pass         = auth_elements[10]
        
        forgot_link_btn       = auth_elements[11]
        forgot_tab            = auth_elements[12]
        auth_tabs             = auth_elements[13]
        forgot_status         = auth_elements[14]
        back_to_signin_btn    = auth_elements[15]
        
        reset_new_password    = auth_elements[16]
        reset_repeat_password = auth_elements[17]
        reset_show_pass       = auth_elements[18]
        reset_save_btn        = auth_elements[19]

    # --- VIEW 2: MAIN DASHBOARD STUDIO WORKSPACE (PRIVATE) ---
    with gr.Column(visible=False) as private_layout:
        ui = create_generator_ui()

    # --- VIEW 3: SECURE PAYMENT GATEWAY PORTAL ---
    with gr.Column(visible=False) as payment_layout:
        payment_container, back_to_workspace_btn = create_payment_ui()

    # --- VIEW 4: CONTACT US SUPPORT PORTAL ---
    with gr.Column(visible=False) as contact_layout:
        contact_container, contact_submit_btn, c_email, c_subject, c_msg, c_status = create_contact_us_ui()


    # ==========================================
    # SYSTEM INTERACTION BACKEND ROUTING LOGIC
    # ==========================================

    # --- SIGN IN CONTROL ---
    def handle_login(username, password):
        msg, success = login_user(username, password)
        if success:
            # Cleanly extracts the real underlying username from database confirmation text
            extracted_name = username.strip()
            return msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), extracted_name
        return msg, gr.update(), gr.update(), gr.update(), gr.update(), "anonymous"

    login_btn.click(
        fn=handle_login,
        inputs=[login_user_input, login_pass],
        outputs=[login_status, public_layout, private_layout, payment_layout, contact_layout, session_user]
    ).then( # Instantly updates the visible gallery window using the freshly mapped identity context
        fn=load_existing_favorites,
        inputs=[session_user],
        outputs=[ui["saved_gallery"]]
    )

    # --- ACCOUNT REGISTRATION CONTROL ---
    def handle_registration(username, email, password, repeat_password):
        msg, success = save_user(username, email, password, repeat_password)
        if success:
            extracted_name = username.strip()
            return msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), extracted_name
        return msg, gr.update(), gr.update(), gr.update(), gr.update(), "anonymous"

    register_btn.click(
        fn=handle_registration,
        inputs=[reg_user, reg_email, reg_pass, reg_repeat_pass],
        outputs=[register_status, public_layout, private_layout, payment_layout, contact_layout, session_user]
    ).then(
        fn=load_existing_favorites,
        inputs=[session_user],
        outputs=[ui["saved_gallery"]]
    )

    # --- PASSWORD RECOVERY EVENTS ---
    forgot_link_btn.click(
        fn=direct_forgot_password_trigger,
        inputs=[login_user_input], 
        outputs=[login_status, forgot_tab, auth_tabs, reset_new_password, reset_repeat_password, reset_save_btn, reset_show_pass]
    )
    
    reset_save_btn.click(
        fn=commit_new_password,
        inputs=[login_user_input, reset_new_password, reset_repeat_password],
        outputs=[forgot_status]
    )

    back_to_signin_btn.click(
        fn=lambda: (gr.update(selected="signin_tab"), gr.update(visible=False)),
        inputs=None,
        outputs=[auth_tabs, forgot_tab]
    )

    # --- MAIN GENERATOR CORE ENGINE EVENT HOOKS ---
    ui["mode"].change(
        fn=update_ui,
        inputs=[ui["mode"]],
        outputs=[ui["sketch_inputs"], ui["sketch_img"], ui["prompt"]]
    )

    ui["generate_btn"].click(
        fn=update_ui,
        inputs=[ui["mode"]],
        outputs=[ui["sketch_inputs"], ui["sketch_img"], ui["prompt"]]
    ).then(
        fn=set_processing_notice,
        inputs=None,
        outputs=[ui["status_message"]]
    ).then(
        fn=generate,
        inputs=[ui["mode"], ui["count_slider"], ui["sketch_img"], ui["prompt"]],
        outputs=[ui["selected_preview"], ui["output_gallery"], ui["status_message"], ui["original_image_backup"]]
    )

    ui["output_gallery"].select(
        fn=on_gallery_select,
        inputs=[ui["output_gallery"]],
        outputs=[ui["selected_preview"], ui["modification_status"], ui["original_image_backup"]]
    )

    ui["submit_modification_btn"].click(
        fn=modify_selected_image,
        inputs=[ui["selected_preview"], ui["modify_input_prompt"], ui["strength_control"], ui["prompt"]],
        outputs=[ui["selected_preview"], ui["modification_status"]]
    )

    ui["reset_original_btn"].click(
        fn=reset_to_original_image,
        inputs=[ui["original_image_backup"]],
        outputs=[ui["selected_preview"], ui["modification_status"]]
    )

    # Passes session_user value directly inside append logic pipeline
    ui["save_favorite_btn"].click(
        fn=append_to_favorites, 
        inputs=[ui["selected_preview"], ui["custom_filename_input"], ui["favorites_cache"], session_user], 
        outputs=[ui["favorites_cache"], ui["saved_gallery"], ui["modification_status"]]
    )

    # Refresh gallery context dynamic window upon workspace tab view changing
    ui["saved_gallery_tab"].select(
        fn=load_existing_favorites,
        inputs=[session_user],
        outputs=[ui["saved_gallery"]]
    )

    # --- SUPPORT PANEL EVENT ROUTING ---
    contact_submit_btn.click(
        fn=handle_contact_submit,
        inputs=[c_email, c_subject, c_msg],
        outputs=[c_status]
    )

    # --- CENTRALIZED DASHBOARD USER DROPDOWN ACTION CONTROLLER ---
    def handle_menu_navigation(choice):
        if choice == "Log out":
            return (
                gr.update(visible=True),   # Return to Login Layout
                gr.update(visible=False),  # Hide Dashboard
                gr.update(visible=False),  # Hide Checkout
                gr.update(visible=False),  # Hide Contact Layout
                "Profile Menu",
                gr.update(value=""),       # Clear sign-in fields
                gr.update(value=""),
                '<div style="color: #4ade80; font-weight: bold; text-align: center; margin-top: 10px;">Logout successful. Session cleared!</div>',
                "anonymous"                # Reset session variable profile context
            )
        elif choice == "Account Settings":
            gr.Info("Redirecting you to our contact team for configuration requests...")
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), "Profile Menu", gr.update(), gr.update(), gr.update(), gr.update()
        elif choice == "Payment":
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "Profile Menu", gr.update(), gr.update(), gr.update(), gr.update()
        
        return gr.update(), gr.update(), gr.update(), gr.update(), "Profile Menu", gr.update(), gr.update(), gr.update(), gr.update()

    ui["user_menu"].change(
        fn=handle_menu_navigation,
        inputs=[ui["user_menu"]],
        outputs=[public_layout, private_layout, payment_layout, contact_layout, ui["user_menu"], login_user_input, login_pass, login_status, session_user]
    )

    # --- SECURE PORTAL RETURN REDIRECT TRACKING ---
    back_to_workspace_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
        inputs=None,
        outputs=[public_layout, private_layout, payment_layout, contact_layout]
    )
    
    # --- FORCE INJECT DYNAMIC GLOBAL MASTER DARK GRAPHICS ENVIRONMENT ---
    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { document.documentElement.classList.add('dark'); }"
    )

if __name__ == "__main__":
    demo.launch(share=True)