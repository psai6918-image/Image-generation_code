import gradio as gr
# Import everything from your two files
from frontend_final import create_login_ui, LOGIN_CSS, save_user, login_user
from image_generation import create_generator_ui, GENERATOR_CSS, generate, on_gallery_select, modify_selected_image, update_ui

# Combine the styles cleanly
COMBINED_CSS = LOGIN_CSS + "\n" + GENERATOR_CSS

with gr.Blocks(css=COMBINED_CSS, title="AI Studio Workspace") as demo:
    generated_cache = gr.State([])

    # --- PUBLIC LAYOUT CONTAINER ---
    with gr.Column(visible=True) as public_layout:
        with gr.Tabs() as main_tabs:
            
            # TAB 1: Gateway Access (Your frontend UI)
            with gr.Tab("Login", id="gateway_tab") as gateway_tab:
                auth_elements = create_login_ui()
                # Unpack the tuple variables exactly as returned from frontend.py
                login_user_input, login_pass, login_btn, login_status = auth_elements[0:4]
                reg_user, reg_email, reg_pass, reg_repeat_pass = auth_elements[4:8]
                birth_month, birth_day, birth_year, register_btn, register_status = auth_elements[8:13]
                show_pass_checkbox = auth_elements[13]
                
            with gr.Tab("Product") as product_tab:
                gr.Markdown("### Product Overview")
            
            with gr.Tab("Demo") as demo_tab:
                gr.Markdown("### Product Demo Video/Info")
            
            with gr.Tab("Pricing") as pricing_tab:
                gr.Markdown("### Pricing Plans")

    # --- PRIVATE LAYOUT CONTAINER (Starts hidden) ---
    with gr.Column(visible=False) as private_layout:
        with gr.Tabs():
            with gr.Tab("AI Image Studio", id="studio_tab") as studio_tab:
                studio_elements = create_generator_ui()
                # Unpack the tuple variables exactly as returned from image_generation.py
                processed_preview, output_gallery, modify_panel, selected_preview = studio_elements[0:4]
                modify_input_prompt, strength_control, submit_modification_btn, modification_output = studio_elements[4:8]
                mode, count_slider, prompt, sketch_inputs, sketch_img, generate_btn, status_message = studio_elements[8:15]
     
   # --- SINGLE CONSOLIDATED LOGIN NAVIGATION HANDLER ---
    def handle_login_navigation(username, password):
        message, is_success = login_user(username, password)
        if is_success:
            return message, gr.update(visible=False), gr.update(visible=True)
        return message, gr.update(), gr.update()

    # --- SINGLE CONSOLIDATED REGISTRATION NAVIGATION HANDLER ---
    def handle_register_navigation(username, email, password, repeat_password, month, day, year):
        # Call the backend logic inside frontend_final.py
        message, is_success = save_user(username, email, password, repeat_password, month, day, year)
        if is_success:
            # Hide the public layout, reveal the private dashboard layout context
            return message, gr.update(visible=False), gr.update(visible=True)
        return message, gr.update(), gr.update()

    # --- ROUTING ACTION MOVEMENT HANDLERS ---
    login_btn.click(
        fn=handle_login_navigation,
        inputs=[login_user_input, login_pass],
        outputs=[login_status, public_layout, private_layout]
    )

    register_btn.click(
        fn=handle_register_navigation,
        inputs=[reg_user, reg_email, reg_pass, reg_repeat_pass, birth_month, birth_day, birth_year],
        outputs=[register_status, public_layout, private_layout]
    )
    
    # --- WIRE UP ALL OTHER EVENT HANDLERS HERE ---
    # Image processing events
    mode.change(fn=update_ui, inputs=mode, outputs=[sketch_inputs, processed_preview, count_slider, prompt])
    generate_btn.click(fn=generate, inputs=[mode, count_slider, sketch_img, prompt], outputs=[processed_preview, output_gallery, status_message, generated_cache])
    output_gallery.select(fn=on_gallery_select, inputs=[generated_cache], outputs=[modify_panel, selected_preview, modification_output])
    submit_modification_btn.click(fn=modify_selected_image, inputs=[selected_preview, modify_input_prompt, strength_control], outputs=[modification_output])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch()