import gradio as gr
from db_config import create_login_ui, LOGIN_CSS, save_user, login_user
# Import all required functions alongside the new structured UI creator
from modified_file import (
    create_generator_ui, GENERATOR_CSS, generate, on_gallery_select, 
    modify_selected_image, update_ui, filter_favorites, set_processing_notice, 
    reset_workspace_image, append_to_favorites
)

COMBINED_CSS = LOGIN_CSS + "\n" + GENERATOR_CSS

with gr.Blocks(css=COMBINED_CSS, title="AI Studio Workspace") as demo:
    generated_cache = gr.State([])

    # --- PUBLIC LAYOUT CONTAINER ---
    with gr.Column(visible=True) as public_layout:
        with gr.Tabs() as main_tabs:
            with gr.Tab("Login", id="gateway_tab") as gateway_tab:
                auth_elements = create_login_ui()
                login_user_input, login_pass, login_btn, login_status = auth_elements[0:4]
                reg_user, reg_email, reg_pass, reg_repeat_pass = auth_elements[4:8]
                register_btn, register_status, show_pass_checkbox = auth_elements[8:11]

    # --- PRIVATE LAYOUT CONTAINER (Starts hidden) ---
    with gr.Column(visible=False) as private_layout:
        # Generate the UI structure dictionary dynamically
        ui = create_generator_ui()

    # --- ROUTING NAVIGATION HANDLERS ---
    def handle_login_navigation(username, password):
        message, is_success = login_user(username, password)
        if is_success:
            return message, gr.update(visible=False), gr.update(visible=True)
        return message, gr.update(), gr.update()

    def handle_register_navigation(username, email, password, repeat_password):
        message, is_success = save_user(username, email, password, repeat_password)
        if is_success:
            return message, gr.update(visible=False), gr.update(visible=True)
        return message, gr.update(), gr.update()

    login_btn.click(
        fn=handle_login_navigation,
        inputs=[login_user_input, login_pass],
        outputs=[login_status, public_layout, private_layout]
    )

    register_btn.click(
        fn=handle_register_navigation,
        inputs=[reg_user, reg_email, reg_pass, reg_repeat_pass],
        outputs=[register_status, public_layout, private_layout]
    )
    
    # --- WORKSPACE EVENT INTERACTION HANDLERS ---
    ui["search_bar"].change(
        fn=filter_favorites,
        inputs=[ui["search_bar"], ui["favorites_cache"]],
        outputs=[ui["saved_gallery"]]
    )

    ui["mode"].change(
        fn=update_ui, 
        inputs=ui["mode"], 
        outputs=[ui["sketch_inputs"], ui["processed_preview"], ui["count_slider"], ui["prompt"]]
    )

    ui["generate_btn"].click(
        fn=set_processing_notice,
        inputs=None,
        outputs=[ui["status_message"], ui["modify_panel"]]
    ).then(
        fn=generate,
        inputs=[ui["mode"], ui["count_slider"], ui["sketch_img"], ui["prompt"]],
        outputs=[ui["processed_preview"], ui["output_gallery"], ui["status_message"], generated_cache]
    )

    ui["output_gallery"].select(
        fn=on_gallery_select,
        inputs=[generated_cache],
        outputs=[ui["modify_panel"], ui["selected_preview"], ui["modification_status"], ui["original_image_backup"]]
    )

    ui["reset_btn"].click(
        fn=reset_workspace_image,
        inputs=[ui["original_image_backup"]],
        outputs=[ui["selected_preview"], ui["modification_status"]]
    )

    ui["save_favorite_btn"].click(
        fn=append_to_favorites,
        inputs=[ui["selected_preview"], ui["custom_filename_input"], ui["favorites_cache"]],
        outputs=[ui["favorites_cache"], ui["saved_gallery"], ui["modification_status"]]
    )

    ui["submit_modification_btn"].click(
        fn=lambda: '<div style="color: #d97706; font-weight: bold;">⏳ Rendering modifications inside workspace window...</div>',
        inputs=None,
        outputs=[ui["modification_status"]]
    ).then(
        fn=modify_selected_image,
        inputs=[ui["selected_preview"], ui["modify_input_prompt"], ui["strength_control"], ui["prompt"]],
        outputs=[ui["selected_preview"], ui["modification_status"]]
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch()