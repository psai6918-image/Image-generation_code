import gradio as gr
# Import everything from your two files
from frontend_final import create_login_ui, LOGIN_CSS, toggle_password_visibility, save_user, login_user
from image_generation import create_generator_ui, GENERATOR_CSS, generate, on_gallery_select, modify_selected_image, update_ui

# Combine the styles cleanly
COMBINED_CSS = LOGIN_CSS + "\n" + GENERATOR_CSS

with gr.Blocks(css=COMBINED_CSS, title="AI Studio Workspace") as demo:
    generated_cache = gr.State([])

    # This is the master Tab controller layout
    with gr.Tabs() as main_tabs:
        
        # TAB 1: Gateway Access (Your frontend UI)
        with gr.Tab("Gateway Access", id="gateway_tab") as gateway_tab:
            auth_elements = create_login_ui()
            # Unpack the tuple variables exactly as returned from frontend.py
            login_user_input, login_pass, login_btn, login_status = auth_elements[0:4]
            reg_user, reg_email, reg_pass, reg_repeat_pass = auth_elements[4:8]
            birth_month, birth_day, birth_year, register_btn, register_status = auth_elements[8:13]
            show_pass_checkbox = auth_elements[13]

        # TAB 2: Protected Workspace (Your image generation UI - starts hidden)
        with gr.Tab("AI Image Studio", id="studio_tab", visible=False) as studio_tab:
            studio_elements = create_generator_ui()
            # Unpack the tuple variables exactly as returned from image_generation.py
            processed_preview, output_gallery, modify_panel, selected_preview = studio_elements[0:4]
            modify_input_prompt, strength_control, submit_modification_btn, modification_output = studio_elements[4:8]
            mode, count_slider, prompt, sketch_inputs, sketch_img, generate_btn, status_message = studio_elements[8:15]

    # --- WIRE UP ALL EVENT HANDLERS HERE ---
    show_pass_checkbox.select(fn=toggle_password_visibility, inputs=show_pass_checkbox, outputs=[login_pass, reg_pass, reg_repeat_pass])
    register_btn.click(fn=save_user, inputs=[reg_user, reg_email, reg_pass, reg_repeat_pass, birth_month, birth_day, birth_year], outputs=register_status)
    
    # The login button handles switching tabs upon success
    # Wrap the login function to return Gradio updates that toggle tabs
    def login_and_switch(username, password):
        result = login_user(username, password)
        # If login_user returns a success message, show studio tab and hide gateway
        success_indicators = ("🔓", "Welcome", "Login successful")
        if any(token in str(result) for token in success_indicators):
            # Make studio tab visible, hide gateway tab and select the studio tab
            return result, gr.update(visible=True), gr.update(visible=False), gr.update(selected=1)
        # otherwise keep showing the gateway and return the status message (select gateway)
        return result, gr.update(visible=False), gr.update(visible=True), gr.update(selected=0)

    login_btn.click(
        fn=login_and_switch,
        inputs=[login_user_input, login_pass],
        outputs=[login_status, studio_tab, gateway_tab, main_tabs],
    )

    # Image processing events
    mode.change(fn=update_ui, inputs=mode, outputs=[sketch_inputs, processed_preview, count_slider, prompt])
    generate_btn.click(fn=generate, inputs=[mode, count_slider, sketch_img, prompt], outputs=[processed_preview, output_gallery, status_message, generated_cache])
    output_gallery.select(fn=on_gallery_select, inputs=[generated_cache], outputs=[modify_panel, selected_preview, modification_output])
    submit_modification_btn.click(fn=modify_selected_image, inputs=[selected_preview, modify_input_prompt, strength_control], outputs=[modification_output])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch()
