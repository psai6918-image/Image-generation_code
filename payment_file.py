import gradio as gr
import re
from datetime import datetime

def process_payment(card_name, card_number, expiry, cvc):
    # Strip spaces and dashes
    card_number = re.sub(r'\s+|-', '', card_number)
    cvc = cvc.strip()
    
    # Basic Validation Checks
    if not card_name.strip():
        return gr.update(value="❌ Error: Cardholder name is required.", visible=True)
        
    if not re.match(r'^\d{13,19}$', card_number):
        return gr.update(value="❌ Error: Invalid card number format (should be 13-19 digits).", visible=True)
        
    if not re.match(r'^(0[1-9]|1[0-2])\/?([0-9]{2})$', expiry):
        return gr.update(value="❌ Error: Invalid expiry format. Use MM/YY.", visible=True)
        
    if not re.match(r'^\d{3,4}$', cvc):
        return gr.update(value="❌ Error: CVC must be 3 or 4 digits.", visible=True)
    
    # Validate Expiry Date is in the future
    try:
        exp_month, exp_year = map(int, expiry.split('/'))
        exp_year += 2000 # Convert YY to YYYY
        current_time = datetime.now()
        if exp_year < current_time.year or (exp_year == current_time.year and exp_month < current_time.month):
            return gr.update(value="❌ Error: The card has expired.", visible=True)
    except ValueError:
        return gr.update(value="❌ Error: Invalid expiry date.", visible=True)

    # --- Backend Processing Goes Here ---
    masked_card = f"•••• •••• •••• {card_number[-4:]}"
    success_msg = f"🎉 Payment processed successfully for {card_name} (Card: {masked_card})!"
    
    return gr.update(value=success_msg, visible=True)

# Building the Custom CSS for a professional checkout feel
custom_css = """
.checkout-box { background-color: #f9fafb; border-radius: 8px; padding: 20px; border: 1px solid #e5e7eb; max-width: 600px; margin: 0 auto; }
.pay-btn { background-color: #2563eb !important; color: white !important; }
.pay-btn:hover { background-color: #1d4ed8 !important; }
"""

with gr.Blocks(css=custom_css, title="Secure Checkout") as demo:
    gr.Markdown("# 💳 Secure Payment Portal", elem_id="title")
    gr.Markdown("Please enter your payment details below to complete your purchase.")
    
    # Removed the second Column (Order Summary) entirely
    with gr.Column(elem_classes="checkout-box"):
        gr.Markdown("### Card Information")
        
        name_input = gr.Textbox(
            label="Cardholder Name", 
            placeholder="John Doe"
        )
        
        num_input = gr.Textbox(
            label="Card Number", 
            placeholder="1234 5678 1234 5678",
            max_lines=1
        )
        
        with gr.Row():
            expiry_input = gr.Textbox(
                label="Expiration Date", 
                placeholder="MM/YY", 
                max_lines=1
            )
            cvc_input = gr.Textbox(
                label="CVC / CVV", 
                placeholder="123", 
                type="password", # Masks the CVC digits for privacy
                max_lines=1
            )
        
        submit_btn = gr.Button("Complete Payment", elem_classes="pay-btn")
        
        # Output Alert Box inside the form container
        status_output = gr.Markdown(visible=False)

    # Wire up the button click event
    submit_btn.click(
        fn=process_payment,
        inputs=[name_input, num_input, expiry_input, cvc_input],
        outputs=status_output
    )

if __name__ == "__main__":
    demo.launch(share=True)
