import gradio as gr
import re
from datetime import datetime

def process_payment(card_name, card_number, expiry, cvc):
    card_number = re.sub(r'\s+|-', '', card_number)
    cvc = cvc.strip()
    
    if not card_name.strip():
        return gr.update(value="❌ Error: Cardholder name is required.", visible=True)
    if not re.match(r'^\d{13,19}$', card_number):
        return gr.update(value="❌ Error: Invalid card number format.", visible=True)
    if not re.match(r'^(0[1-9]|1[0-2])\/?([0-9]{2})$', expiry):
        return gr.update(value="❌ Error: Invalid expiry format. Use MM/YY.", visible=True)
    if not re.match(r'^\d{3,4}$', cvc):
        return gr.update(value="❌ Error: CVC must be 3 or 4 digits.", visible=True)
    
    try:
        exp_month, exp_year = map(int, expiry.split('/'))
        exp_year += 2000
        current_time = datetime.now()
        if exp_year < current_time.year or (exp_year == current_time.year and exp_month < current_time.month):
            return gr.update(value="❌ Error: The card has expired.", visible=True)
    except ValueError:
        return gr.update(value="❌ Error: Invalid expiry date.", visible=True)
        
    return gr.update(value="🎉 Payment Successful! Your Premium features have been unlocked.", visible=True)

def create_payment_ui():
    """Generates a perfect double-column checkout layout."""
    with gr.Group() as payment_container:
        gr.Markdown("# 💳 Secure Payment Portal", elem_id="title")
        gr.Markdown("Please review your order summary and complete your purchase securely below.")
        
        # Side-by-side Layout Wrapper
        with gr.Row(elem_id="checkout-layout-row"):
            
            # LEFT SIDE PANEL: Order Details
            with gr.Column(elem_classes="summary-box", scale=1):
                gr.Markdown("### 📦 Order Summary")
                gr.Markdown(
                    """
                    **Product:** AI Generation Engine Premium Pass  
                    **Billing Term:** Monthly Membership subscription  
                    
                    ---
                    
                    ### 💰 Pricing Breakdown
                    * Subtotal: **$19.99 USD** * VAT / Taxes: **$0.00 USD** **Total Amount Due:** <span style='font-size: 1.2em; color: #38bdf8;'>$19.99 USD</span>
                    """
                )
            
            # RIGHT SIDE PANEL: Credit Card inputs
            with gr.Column(elem_classes="checkout-box", scale=1):
                gr.Markdown("### 💳 Card Information")
                
                name_input = gr.Textbox(label="Cardholder Name", placeholder="John Doe")
                num_input = gr.Textbox(label="Card Number", placeholder="1234 5678 1234 5678", max_lines=1)
                
                with gr.Row():
                    expiry_input = gr.Textbox(label="Expiration Date", placeholder="MM/YY", max_lines=1)
                    cvc_input = gr.Textbox(label="CVC / CVV", placeholder="123", type="password", max_lines=1)
                
                submit_btn = gr.Button("Complete Payment", elem_classes="pay-btn")
                status_output = gr.Markdown(visible=False)
        
        # Clear navigation boundary below the columns
        with gr.Row():
            back_to_workspace_btn = gr.Button("← Return to Workspace", variant="secondary")

    # Wire inner button functionality
    submit_btn.click(
        fn=process_payment,
        inputs=[name_input, num_input, expiry_input, cvc_input],
        outputs=[status_output]
    )

    return payment_container, back_to_workspace_btn