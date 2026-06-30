# pages/contact_us.py
import os
import gradio as gr

# --- DYNAMIC CSS LOADING ---
css_path = os.path.join("assets", "contact_us.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        CONTACT_US_CSS = f.read()
else:
    CONTACT_US_CSS = ""


def create_contact_us_ui():
    """
    Builds the production-ready contact form layout wrapper.
    """
    with gr.Group() as contact_layout:
        with gr.Row(elem_classes=["contact-form-card"]):
            
            with gr.Column(scale=2):
                gr.Markdown("## 📩 Get in Touch")
                gr.Markdown("Have questions regarding your workspace, billing updates, or custom enterprise solutions? Fill out the form below.")
                
                contact_email = gr.Textbox(label="Your Email Address", placeholder="name@domain.com", max_lines=1)
                contact_subject = gr.Textbox(label="Inquiry Subject", placeholder="e.g., Enterprise API Access", max_lines=1)
                contact_message = gr.Textbox(label="Detailed Message", placeholder="Type your message here...", lines=5)
                send_btn = gr.Button("Submit Message", variant="primary")
                contact_status = gr.Markdown("")
            
            with gr.Column(scale=1):
                gr.Markdown("### Direct Support")
                gr.Markdown("📧 **Email:** [contact@aimultimode.com](mailto:contact@aimultimode.com)")
                gr.Markdown("⏱️ **Response Time:** Expect a human response within 24-48 business hours.")

    return contact_layout, send_btn, contact_email, contact_subject, contact_message, contact_status


def handle_contact_submit(email, subject, message):
    """
    Standard input handler for validating and routing the contact message.
    """
    if not email.strip() or not message.strip():
        return "❌ Form incomplete. Please provide both your Email and Message."
    
    # Place your live production mailer trigger (e.g., smtplib) here
    return "✅ Message routed successfully! Our team will get back to you shortly."


# --- Standalone Execution Pipeline ---
if __name__ == "__main__":
    with gr.Blocks(css=CONTACT_US_CSS, title="Contact Test Isolation Environment") as demo:
        # Layout unpacking mirrors app.py orchestration pipeline
        layout, btn, email, sub, msg, status = create_contact_us_ui()
        
        btn.click(
            fn=handle_contact_submit,
            inputs=[email, sub, msg],
            outputs=[status]
        )
        
    demo.launch()