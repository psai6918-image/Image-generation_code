import gradio as gr
# Import everything from your two files
from frontend_final import create_login_ui, LOGIN_CSS, toggle_password_visibility, save_user, login_user
from image_generation import create_generator_ui, GENERATOR_CSS, generate, on_gallery_select, modify_selected_image, update_ui

# Combine the styles cleanly
COMBINED_CSS = LOGIN_CSS + "\n" + GENERATOR_CSS

with gr.Blocks(css=COMBINED_CSS, title="AI Studio Workspace") as demo:
    generated_cache = gr.State([])

    # This is the master Tab controller layout
    with gr.Tabs(visible=True) as main_tabs:
        
        # TAB 1: Gateway Access (Your frontend UI)
        with gr.Tab("Login", id="gateway_tab") as gateway_tab:
            auth_elements = create_login_ui()
            # Unpack the tuple variables exactly as returned from frontend.py
            login_user_input, login_pass, login_btn, login_status = auth_elements[0:4]
            reg_user, reg_email, reg_pass, reg_repeat_pass = auth_elements[4:8]
            birth_month, birth_day, birth_year, register_btn, register_status = auth_elements[8:13]
            show_pass_checkbox = auth_elements[13]
            
        with gr.Tab("Product") as product_tab:
            pass
        
        with gr.Tab("Demo") as demo_tab:
            pass
        
        with gr.Tab("Pricing") as pricing_tab:
            pass 

    # FIXED: Indentation corrected so it is inside 'with gr.Blocks()' but outside 'with gr.Tabs()'
    with gr.Tab("AI Image Studio", id="studio_tab", visible=False) as studio_tab:
        studio_elements = create_generator_ui()
        # Unpack the tuple variables exactly as returned from image_generation.py
        processed_preview, output_gallery, modify_panel, selected_preview = studio_elements[0:4]
        modify_input_prompt, strength_control, submit_modification_btn, modification_output = studio_elements[4:8]
        mode, count_slider, prompt, sketch_inputs, sketch_img, generate_btn, status_message = studio_elements[8:15]
    
    # Combined single clean login function 
    def handle_login_navigation(username, password):
        message, is_success = login_user(username, password)
        
        if is_success:
            # Hide public tabs system, reveal private studio dashboard, select studio tab
            return message, gr.update(visible=False), gr.update(visible=True)
        
        # If login fails, change absolutely nothing about page display states
        return message, gr.update(), gr.update()

    # --- ROUTING ACTION MOVEMENT HANDLER ---
    login_btn.click(
        fn=handle_login_navigation,
        inputs=[login_user_input, login_pass],
        outputs=[login_status, main_tabs, studio_tab]
    )

    # --- WIRE UP ALL EVENT HANDLERS HERE ---
    show_pass_checkbox.select(fn=toggle_password_visibility, inputs=show_pass_checkbox, outputs=[login_pass, reg_pass, reg_repeat_pass])
    register_btn.click(fn=save_user, inputs=[reg_user, reg_email, reg_pass, reg_repeat_pass, birth_month, birth_day, birth_year], outputs=register_status)
    
    # Image processing events
    mode.change(fn=update_ui, inputs=mode, outputs=[sketch_inputs, processed_preview, count_slider, prompt])
    generate_btn.click(fn=generate, inputs=[mode, count_slider, sketch_img, prompt], outputs=[processed_preview, output_gallery, status_message, generated_cache])
    output_gallery.select(fn=on_gallery_select, inputs=[generated_cache], outputs=[modify_panel, selected_preview, modification_output])
    submit_modification_btn.click(fn=modify_selected_image, inputs=[selected_preview, modify_input_prompt, strength_control], outputs=[modification_output])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch()
