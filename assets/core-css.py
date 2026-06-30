/* ============================================================
   1. GLOBAL CANVAS & BACKDROP (THEME FOUNDATION)
============================================================ */
html, body, grad-app, .gradio-container {
    background: 
        radial-gradient(circle at 15% 50%, rgba(236, 72, 153, 0.40), transparent 50%),
        radial-gradient(circle at 85% 30%, rgba(56, 189, 248, 0.40), transparent 50%),
        radial-gradient(circle at 50% 90%, rgba(139, 92, 246, 0.40), transparent 50%),
        linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #020617 100%) !important;
    background-attachment: fixed !important;
    min-height: 100vh !important;
}

.gradio-container {
    padding: 30px !important;
}

/* Base structural cleanup - Removes block nesting background artifacts */
.gradio-container .block,
.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container .group,
.gradio-container .gr-group,
.gradio-container div[class*="svelte-"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Flex layout standardizations */
.gradio-container div[class*="row"], .gradio-container .row {
    display: flex !important;
    background: transparent !important;
    border: none !important;
}

.gradio-container div[class*="column"], .gradio-container .column {
    display: flex !important;
    flex-direction: column !important;
    background: transparent !important;
    border: none !important;
}

/* ============================================================
   2. GRADIO NATIVE COMPONENT VARIABLE OVERRIDES
============================================================ */
:root, .gradio-container {
    --block-background-fill: rgba(0, 0, 0, 0.4) !important;
    --block-border-color: rgba(255, 255, 255, 0.15) !important;
    --input-background-fill: rgba(0, 0, 0, 0.5) !important;
    --input-border-color: rgba(255, 255, 255, 0.15) !important;
    --body-text-color: #ffffff !important;
    --body-text-color-subdued: rgba(255, 255, 255, 0.6) !important;
    --block-label-text-color: #ffffff !important;
}

/* ============================================================
   3. TYPOGRAPHY & TEXT ACCENTS (MAX CONTRAST)
============================================================ */
.gradio-container h2, 
.gradio-container .prose h2, 
.gradio-container div[class*="markdown"] h2 {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 26px !important;
}

.gradio-container label span, 
.gradio-container .text-sm,
.gradio-container p,
.gradio-container span,
.gradio-container label,
.gradio-container .block-title,
.gradio-container .block-label {
    color: #ffffff !important;
    font-weight: 600 !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4) !important;
}

/* ============================================================
   4. FORM INPUTS & INTERACTIONS
============================================================ */
.gradio-container input, 
.gradio-container textarea, 
.gradio-container select,
.gradio-container div[class*="input"] {
    background-color: rgba(0, 0, 0, 0.5) !important;
    background: rgba(0, 0, 0, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}

input::placeholder, textarea::placeholder {
    color: rgba(255, 255, 255, 0.35) !important;
}

/* Focus State updates */
.gradio-container input:focus, 
.gradio-container textarea:focus {
    border-color: rgba(236, 72, 153, 0.7) !important;
    background-color: rgba(255, 255, 255, 0.22) !important;
}

/* ============================================================
   5. PRIMARY INTERACTIVE BUTTON SIGNATURE
============================================================ */
.gradio-container button.primary, 
.gradio-container button[class*="primary"] {
    background: linear-gradient(90deg, #ec4899, #8b5cf6) !important;
    background-image: linear-gradient(90deg, #ec4899, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 14px 36px !important;
    cursor: pointer !important;
    width: 100% !important; 
    display: block !important;
    margin: 20px auto 0 auto !important;
    box-shadow: 0 8px 20px rgba(236, 72, 153, 0.4) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
}

.gradio-container button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 12px 24px rgba(236, 72, 153, 0.55) !important;
}

/* ============================================================
   6. UTILITIES (CLEANUP)
============================================================ */
footer { 
    display: none !important; 
}