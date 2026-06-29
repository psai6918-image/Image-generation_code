import gradio as gr
from db_config import (
    create_login_ui, LOGIN_CSS, save_user, login_user,
    direct_forgot_password_trigger, commit_new_password
)
# Fixed import mapping to pull layout properties from sample_file directly
from samples_file import (
    create_generator_ui, GENERATOR_CSS, generate, on_gallery_select, 
    modify_selected_image, update_ui, set_processing_notice, 
    append_to_favorites, reset_to_original_image, clear_workspace_preview
)

COMBINED_CSS = LOGIN_CSS + "\n" + GENERATOR_CSS

with gr.Blocks(css=COMBINED_CSS, title="AI Studio Workspace") as demo:
    
    # --- AUTHENTICATION LAYOUT ---
    with gr.Column(visible=True) as public_layout:
        auth_elements = create_login_ui()
        
        # Exact matching unpacking index arrays for the 20 components returned from db_config
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

    # --- GENERATOR UI LAYOUT ---
    with gr.Column(visible=False) as private_layout:
        ui = create_generator_ui()

    # --- MAIN LOGIN ROUTER LOGIC ---
    def handle_login(username, password):
        msg, success = login_user(username, password)
        if success:
            # Transition out the login layout panel and swap visibility targets
            return msg, gr.update(visible=False), gr.update(visible=True)
        else:
            return msg, gr.update(), gr.update()

    # Wire Up Account Registration Actions
    register_btn.click(
        fn=save_user,
        inputs=[reg_user, reg_email, reg_pass, reg_repeat_pass],
        outputs=[register_status]
    )

    # Wire Up Sign In Routing (Swaps panels on success)
    login_btn.click(
        fn=handle_login, 
        inputs=[login_user_input, login_pass], 
        outputs=[login_status, public_layout, private_layout]
    )

    # --- PASSWORD RECOVERY LOGIC WIRE-UP ---
    # Trigger account lookup when user clicks "Forgot Password?"
    forgot_link_btn.click(
        fn=direct_forgot_password_trigger,
        inputs=[login_user_input], 
        outputs=[login_status, forgot_tab, auth_tabs, reset_new_password, reset_repeat_password, reset_save_btn, reset_show_pass]
    )
    
    # Save the updated cryptographic credentials to the database schema
    reset_save_btn.click(
        fn=commit_new_password,
        inputs=[login_user_input, reset_new_password, reset_repeat_password],
        outputs=[forgot_status]
    )

    # Clean backtracking function to reset visibility thresholds and return back home
    back_to_signin_btn.click(
        fn=lambda: (gr.update(selected="signin_tab"), gr.update(visible=False)),
        inputs=None,
        outputs=[auth_tabs, forgot_tab]
    )

    # --- EVENT WIRING FOR THE DASHBOARD GENERATOR ---
    ui["mode"].change(update_ui, inputs=[ui["mode"]], outputs=[ui["sketch_inputs"], ui["count_slider"], ui["prompt"]])

    ui["generate_btn"].click(
        fn=clear_workspace_preview,
        inputs=None,
        outputs=[ui["selected_preview"]]
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

    ui["save_favorite_btn"].click(
        fn=append_to_favorites, 
        inputs=[ui["selected_preview"], ui["custom_filename_input"], ui["favorites_cache"]], 
        outputs=[ui["favorites_cache"], ui["saved_gallery"], ui["modification_status"]]
    )

if __name__ == "__main__":
    demo.queue().launch()