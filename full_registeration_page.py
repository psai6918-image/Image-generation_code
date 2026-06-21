import os
import csv
import torch
import numpy as np
import cv2
import uuid
import secrets
import hashlib
import threading
from datetime import datetime
from PIL import Image
import mysql.connector
from mysql.connector import Error
import gradio as gr
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline

# Speed up matrix multiplication on compatible GPUs
torch.backends.cuda.matmul.allow_tf32 = True

# --- 1. DATABASE CONFIGURATION ---
DB_CONFIG = {
    'host': 'localhost',
    'database': 'dbo',
    'user': 'root',
    'password': 'NameisRoot909'
}

def init_db():
    """Ensures the users table exists in the MySQL database upon startup."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Username VARCHAR(255) NOT NULL UNIQUE,
        Email VARCHAR(255) NOT NULL UNIQUE,
        DOB DATE NULL,
        Password VARCHAR(255) NOT NULL
    );
    """
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute(create_table_query)
            connection.commit()
            print("MySQL Table verified successfully.")
    except Error as e:
        print(f"Error during database initialization: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# --- 2. SECURITY & HASHING FUNCTIONS ---
def hash_password(password: str) -> str:
    """Generates a secure 16-byte random salt and hashes the password using SHA-256."""
    salt = secrets.token_hex(16)
    hashed_bytes = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    return f"{salt}:{hashed_bytes}"

# --- 3. MODEL CONFIGURATION & INITIALIZATION ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_LOG_FILE = "generation_logs.csv"

gpu_lock = threading.Lock()
log_lock = threading.Lock()

if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Mode", "Prompt", "Num Images Request", "Saved File Names"])

print(f"Loading Models on {device}...")
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=dtype
).to(device)

pipe_text2img = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID, torch_dtype=dtype, safety_checker=None
).to(device)

pipe_sketch2img = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL_ID, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
).to(device)

if device == "cuda":
    pipe_text2img.enable_model_cpu_offload()
    pipe_sketch2img.enable_model_cpu_offload()
else:
    pipe_text2img.enable_attention_slicing()
    pipe_sketch2img.enable_attention_slicing()

print("Models Loaded Successfully!")

# --- 4. INTERNATIONALIZATION (I18N) DICTIONARY ---
LANGUAGES = {
    "English": {
        "welcome_title": "# 🚀 Welcome to Our Innovation Platform",
        "welcome_subtitle": "### Experience the power of next-generation AI tools built just for you.",
        "why_join_title": "## Why Join Us?",
        "why_join_text": "* **Instant Access:** Build, test, and deploy with top-tier AI pipelines.\n* **Cloud-Powered:** Heavy computational lift handled seamlessly.",
        "features_preview": "### Features Preview",
        "create_account": "## Create Your Free Account",
        "username": "Username",
        "email": "Email Address",
        "password": "Password",
        "confirm_password": "Confirm Password",
        "register_btn": "Register Now",
        "show_btn": "👁️ Show",
        "hide_btn": "🙈 Hide",
        "studio_title": "# 🎨 AI Image Generation Studio (Batch Mode)",
        "inspo_stream": "### 🐶 Live Inspo Stream 🐱",
        "select_mode": "1. Select Mode",
        "prompt": "Prompt",
        "num_images": "Number of Images to Generate (Parallel Batch)",
        "upload_sketch": "Upload or Draw Sketch",
        "generate_btn": "Generate Batch",
        "preview_label": "Processed Edge Map Preview",
        "gallery_label": "Generated Output Images",
        "err_fields": "### ⚠️ All fields are required!",
        "err_match": "### ❌ Passwords do not match\nPlease verify both fields.",
        "err_user_taken": "### ❌ Username is already taken.",
        "err_email_taken": "### ❌ This email is already registered.",
        "success_reg": "### 🎉 Welcome aboard!\nRedirecting to studio pipeline..."
    },
    "Telugu (తెలుగు)": {
        "welcome_title": "# 🚀 మా ఇన్నోవేషన్ ప్లాట్‌ఫారమ్‌కు స్వాగతం",
        "welcome_subtitle": "### మీ కోసమే ప్రత్యేకంగా రూపొందించిన నెక్స్ట్ జనరేషన్ AI సాధనాల శక్తిని అనుభవించండి.",
        "why_join_title": "## మాతో ఎందుకు చేరాలి?",
        "why_join_text": "* **తక్షణ యాక్సెస్:** అగ్రశ్రేణి AI పైప్‌లైన్‌లతో నిర్మించండి, పరీక్షించండి మరియు అమలు చేయండి.\n* **క్లౌడ్-పవర్డ్:** భారీ కంప్యూటేషనల్ లోడ్ సజావుగా నిర్వహించబడుతుంది.",
        "features_preview": "### ఫీచర్స్ ప్రివ్యూ",
        "create_account": "## మీ ఉచిత ఖాతాను సృష్టించండి",
        "username": "వినియోగదారు పేరు (Username)",
        "email": "ఈమెయిల్ చిరునామా",
        "password": "పాస్‌వర్డ్",
        "confirm_password": "పాస్‌వర్డ్‌ను నిర్ధారించండి",
        "register_btn": "ఇప్పుడే నమోదు చేసుకోండి",
        "show_btn": "👁️ చూపించు",
        "hide_btn": "🙈 దాచు",
        "studio_title": "# 🎨 AI ఇమేజ్ జనరేషన్ స్టూడియో (బ్యాచ్ మోడ్)",
        "inspo_stream": "### 🐶 లైవ్ ఇన్స్పో స్ట్రీమ్ 🐱",
        "select_mode": "1. మోడ్‌ను ఎంచుకోండి",
        "prompt": "ప్రాంప్ట్ (Prompt)",
        "num_images": "సృష్టించవలసిన చిత్రాల సంఖ్య",
        "upload_sketch": "స్కెచ్ అప్‌లోడ్ చేయండి లేదా గీయండి",
        "generate_btn": "బ్యాచ్ జనరేట్ చేయండి",
        "preview_label": "ప్రాసెస్ చేయబడిన ఎడ్జ్ మ్యాప్ ప్రివ్యూ",
        "gallery_label": "సృష్టించబడిన అవుట్‌పుట్ చిత్రాలు",
        "err_fields": "### ⚠️ అన్ని ఫీల్డ్‌లు తప్పనిసరి!",
        "err_match": "### ❌ పాస్‌వర్డ్‌లు సరిపోలడం లేదు\nదయచేసి రెండు ఫీల్డ్‌లను ధృవీకరించండి.",
        "err_user_taken": "### ❌ ఈ వినియోగదారు పేరు ఇప్పటికే తీసుకోబడింది.",
        "err_email_taken": "### ❌ ఈ ఈమెయిల్ ఇప్పటికే నమోదై ఉంది.",
        "success_reg": "### 🎉 స్వాగతం!\nస్టూడియో పైప్‌లైన్‌కు మళ్లించబడుతోంది..."
    },
    "Tamil (தமிழ்)": {
        "welcome_title": "# 🚀 எங்களது கண்டுபிடிப்பு தளத்திற்கு வரவேற்கிறோம்",
        "welcome_subtitle": "### உங்களுக்காகவே உருவாக்கப்பட்ட அடுத்த தலைமுறை AI கருவிகளின் ஆற்றலை அனுபவியுங்கள்.",
        "why_join_title": "## ஏன் எங்களுடன் இணைய வேண்டும்?",
        "why_join_text": "* **உடனடி அணுகல்:** சிறந்த AI பைப்லைன்களைக் கொண்டு உருவாக்குங்கள், சோதியுங்கள் மற்றும் செயல்படுத்துங்கள்.\n* **கிளவுட் ஆற்றல்:** கனமான கணக்கீட்டு பணிகள் தடையின்றி கையாளப்படுகின்றன.",
        "features_preview": "### அம்சங்களின் முன்னோட்டம்",
        "create_account": "## உங்கள் இலவச கணக்கை உருவாக்குங்கள்",
        "username": "பயனர் பெயர் (Username)",
        "email": "மின்னஞ்சல் முகவரி",
        "password": "கடவுச்சொல்",
        "confirm_password": "கடவுச்சொல்லை உறுதிப்படுத்தவும்",
        "register_btn": "இப்போதே பதிவு செய்யவும்",
        "show_btn": "👁️ காட்டு",
        "hide_btn": "🙈 மறை",
        "studio_title": "# 🎨 AI பட உருவாக்க ஸ்டுடியோ (தொகுதி முறை)",
        "inspo_stream": "### 🐶 நேரடி உத்வேக ஓட்டம் 🐱",
        "select_mode": "1. பயன்முறையைத் தேர்ந்தெடுக்கவும்",
        "prompt": "ப்ராம்ப்ட் (Prompt)",
        "num_images": "உருவாக்க வேண்டிய படங்களின் எண்ணிக்கை",
        "upload_sketch": "ஸ்கெட்ச் பதிவேற்றவும் அல்லது வரையவும்",
        "generate_btn": "தொகுதியை உருவாக்கு",
        "preview_label": "செயலாக்கப்பட்ட விளிம்பு வரைபட முன்னோட்டம்",
        "gallery_label": "உருவாக்கப்பட்ட வெளியீட்டு படங்கள்",
        "err_fields": "### ⚠️ அனைத்து புலங்களும் கட்டாயமாகும்!",
        "err_match": "### ❌ கடவுச்சொற்கள் பொருந்தவில்லை\nஇரு புலங்களையும் சரிபார்க்கவும்.",
        "err_user_taken": "### ❌ இந்த பயனர் பெயர் ஏற்கனவே எடுக்கப்பட்டுள்ளது.",
        "err_email_taken": "### ❌ இந்த மின்னஞ்சல் ஏற்கனவே பதிவு செய்யப்பட்டுள்ளது.",
        "success_reg": "### 🎉 வரவேற்பு!\nஸ்டுடியோ பைப்லைனுக்கு திருப்பி விடப்படுகிறது..."
    },
    "Hindi (हिन्दी)": {
        "welcome_title": "# 🚀 हमारे इनोवेशन प्लेटफॉर्म पर आपका स्वागत है",
        "welcome_subtitle": "### केवल आपके लिए बनाए गए अगली पीढ़ी के AI टूल की शक्ति का अनुभव करें.",
        "why_join_title": "## हमसे क्यों जुड़ें?",
        "why_join_text": "* **त्वरित पहुंच:** शीर्ष-स्तरीय AI पाइपलाइनों के साथ बनाएं, परीक्षण करें और तैनात करें.\n* **क्लाउड-पावर्ड:** भारी कम्प्यूटेशनल लोड को आसानी से संभाला जाता है.",
        "features_preview": "### सुविधाओं की झलक",
        "create_account": "## अपना निःशुल्क खाता बनाएं",
        "username": "उपयोगकर्ता नाम",
        "email": "ईमेल पता",
        "password": "पासवर्ड",
        "confirm_password": "पासवर्ड की पुष्टि करें",
        "register_btn": "अभी पंजीकरण करें",
        "show_btn": "👁️ दिखाएं",
        "hide_btn": "🙈 छुपाएं",
        "studio_title": "# 🎨 AI इमेज जनरेशन स्टूडियो (बैच मोड)",
        "inspo_stream": "### 🐶 लाइव इंस्पिरेशन स्ट्रीम 🐱",
        "select_mode": "1. मोड चुनें",
        "prompt": "प्रॉम्ट",
        "num_images": "उत्पन्न करने के लिए छवियों की संख्या",
        "upload_sketch": "स्कैच अपलोड करें या बनाएं",
        "generate_btn": "बैच उत्पन्न करें",
        "preview_label": "प्रसंस्कृत एज मैप पूर्वावलोकन",
        "gallery_label": "उत्पन्न आउटपुट छवियां",
        "err_fields": "### ⚠️ सभी फ़ील्ड आवश्यक हैं!",
        "err_match": "### ❌ पासवर्ड मेल नहीं खाते\nकृपया दोनों फ़ील्ड सत्यापित करें.",
        "err_user_taken": "### ❌ यह उपयोगकर्ता नाम पहले से ही लिया जा चुका है.",
        "err_email_taken": "### ❌ यह ईमेल पहले से ही पंजीकृत है.",
        "success_reg": "### 🎉 आपका स्वागत है!\nस्टूडियो पाइपलाइन पर रीडायरेक्ट किया जा रहा है..."
    },
    "Spanish (Español)": {
        "welcome_title": "# 🚀 Bienvenido a Nuestra Plataforma de Innovación",
        "welcome_subtitle": "### Experimente el poder de las herramientas de IA de próxima generación creadas solo para usted.",
        "why_join_title": "## ¿Por qué unirse?",
        "why_join_text": "* **Acceso instantáneo:** Cree, pruebe e impleme con canalizaciones de IA de primer nivel.\n* **Potenciado por la nube:** Carga computacional pesada gestionada sin problemas.",
        "features_preview": "### Vista previa de funciones",
        "create_account": "## Cree su cuenta gratuita",
        "username": "Nombre de usuario",
        "email": "Correo electrónico",
        "password": "Contraseña",
        "confirm_password": "Confirmar contraseña",
        "register_btn": "Registrarse ahora",
        "show_btn": "👁️ Mostrar",
        "hide_btn": "🙈 Ocultar",
        "studio_title": "# 🎨 Studio de Generación de Imágenes IA (Modo por Lotes)",
        "inspo_stream": "### 🐶 Flujo de Inspiración en Vivo 🐱",
        "select_mode": "1. Seleccionar modo",
        "prompt": "Indicación (Prompt)",
        "num_images": "Número de imágenes a generar",
        "upload_sketch": "Subir o dibujar boceto",
        "generate_btn": "Generar lote",
        "preview_label": "Vista previa del mapa de bordes procesado",
        "gallery_label": "Imágenes de salida generadas",
        "err_fields": "### ⚠️ ¡Todos los campos son obligatorios!",
        "err_match": "### ❌ Las contraseñas no coinciden",
        "err_user_taken": "### ❌ El nombre de usuario ya está en uso.",
        "err_email_taken": "### ❌ Este correo electrónico ya está registrado.",
        "success_reg": "### 🎉 ¡Bienvenido a bordo!\nRedirigiendo al estudio..."
    },
    "French (Français)": {
        "welcome_title": "# 🚀 Bienvenue sur notre plateforme d'innovation",
        "welcome_subtitle": "### Découvrez la puissance des outils IA de nouvelle génération conçus pour vous.",
        "why_join_title": "## Pourquoi nous rejoindre ?",
        "why_join_text": "* **Accès instantané:** Créez, testez et déployez avec des pipelines IA de pointe.\n* **Propulsé par le cloud:** Charges de calcul lourdes gérées de manière transparente.",
        "features_preview": "### Aperçu des fonctionnalités",
        "create_account": "## Créez votre compte gratuit",
        "username": "Nom d'utilisateur",
        "email": "Adresse e-mail",
        "password": "Mot de passe",
        "confirm_password": "Confirmer le mot de passe",
        "register_btn": "S'inscrire maintenant",
        "show_btn": "👁️ Afficher",
        "hide_btn": "🙈 Masquer",
        "studio_title": "# 🎨 Studio de génération d'images IA (Mode batch)",
        "inspo_stream": "### 🐶 Flux d'inspiration en direct 🐱",
        "select_mode": "1. Sélectionner le mode",
        "prompt": "Invite (Prompt)",
        "num_images": "Nombre d'images à générer",
        "upload_sketch": "Téléverser ou dessiner un croquis",
        "generate_btn": "Générer le lot",
        "preview_label": "Aperçu de la carte des contours",
        "gallery_label": "Images de sortie générées",
        "err_fields": "### ⚠️ Tous les champs sont obligatoires !",
        "err_match": "### ❌ Les mots de passe ne correspondent pas",
        "err_user_taken": "### ❌ Ce nom d'utilisateur est déjà pris.",
        "err_email_taken": "### ❌ Cet e-mail est déjà enregistré.",
        "success_reg": "### 🎉 Bienvenue à bord !\nRedirection vers le studio..."
    },
    "German (Deutsch)": {
        "welcome_title": "# 🚀 Willkommen auf unserer Innovationsplattform",
        "welcome_subtitle": "### Erleben Sie die Kraft von KI-Tools der nächsten Generation, die für Sie entwickelt wurden.",
        "why_join_title": "## Warum uns beitreten?",
        "why_join_text": "* **Sofortiger Zugriff:** Erstellen, testen und bereitstellen mit erstklassigen KI-Pipelines.\n* **Cloud-Powered:** Schwere Rechenlasten werden nahtlos bewältigt.",
        "features_preview": "### Vorschau der Funktionen",
        "create_account": "## Erstellen Sie Ihr kostenloses Konto",
        "username": "Benutzername",
        "email": "E-Mail-Adresse",
        "password": "Passwort",
        "confirm_password": "Passwort bestätigen",
        "register_btn": "Jetzt registrieren",
        "show_btn": "👁️ Anzeigen",
        "hide_btn": "🙈 Ausblenden",
        "studio_title": "# 🎨 KI-Bildgenerierungsstudio (Batch-Modus)",
        "inspo_stream": "### 🐶 Live-Inspirationsstream 🐱",
        "select_mode": "1. Modus auswählen",
        "prompt": "Prompt",
        "num_images": "Anzahl der zu generierenden Bilder",
        "upload_sketch": "Skizze hochladen oder zeichnen",
        "generate_btn": "Batch generieren",
        "preview_label": "Verarbeitete Kantenkarten-Vorschau",
        "gallery_label": "Generierte Ausgabebilder",
        "err_fields": "### ⚠️ Alle Felder sind Pflichtfelder!",
        "err_match": "### ❌ Passwörter stimmen nicht überein",
        "err_user_taken": "### ❌ Benutzername ist bereits vergeben.",
        "err_email_taken": "### ❌ Diese E-Mail ist bereits registriert.",
        "success_reg": "### 🎉 Willkommen an Bord!\nWeiterleitung zum Studio..."
    },
    "Mandarin (中文)": {
        "welcome_title": "# 🚀 欢迎来到我们的创新平台",
        "welcome_subtitle": "### 体验专为您量身定制 division 的下一代 AI 工具的力量。",
        "why_join_title": "## 为什么加入我们？",
        "why_join_text": "* **即时访问:** 使用顶级的 AI 流水线进行构建、测试和部署。\n* **云端驱动:** 无缝处理沉重的计算负载。",
        "features_preview": "### 功能预览",
        "create_account": "## 创建您的免费账户",
        "username": "用户名",
        "email": "电子邮箱",
        "password": "密码",
        "confirm_password": "确认密码",
        "register_btn": "立即注册",
        "show_btn": "👁️ 显示",
        "hide_btn": "🙈 隐藏",
        "studio_title": "# 🎨 AI 图像生成工作室 ( Feldman 批量模式)",
        "inspo_stream": "### 🐶 灵感直播流 🐱",
        "select_mode": "1. 选择模式",
        "prompt": "提示词 (Prompt)",
        "num_images": "生成的图像数量",
        "upload_sketch": "上传或绘制草图",
        "generate_btn": "批量生成",
        "preview_label": "处理后的边缘图预览",
        "gallery_label": "生成的输出图像",
        "err_fields": "### ⚠️ 所有字段均为必填项！",
        "err_match": "### ❌ 密码不匹配",
        "err_user_taken": "### ❌ 用户名已被占用。",
        "err_email_taken": "### ❌ 该邮箱已被注册。",
        "success_reg": "### 🎉 欢迎加入！\n正在重定向至工作室..."
    },
    "Japanese (日本語)": {
        "welcome_title": "# 🚀 イノベーションプラットフォームへようこそ",
        "welcome_subtitle": "### あなたのために構築された次世代AIツールのパワーを体験してください。",
        "why_join_title": "## 会員登録のメリット",
        "why_join_text": "* **即時アクセス:** 最高峰のAIパイプラインで構築、テスト、展開。\n* **クラウド駆動:** 重い計算処理もシームレスに処理されます。",
        "features_preview": "### 機能のプレビュー",
        "create_account": "## 無料アカウントを作成する",
        "username": "ユーザー名",
        "email": "メールアドレス",
        "password": "パスワード",
        "confirm_password": "パスワードの確認",
        "register_btn": "今すぐ登録する",
        "show_btn": "👁️ 表示",
        "hide_btn": "🙈 非表示",
        "studio_title": "# 🎨 AI画像生成スタジオ（バッチモード）",
        "inspo_stream": "### 🐶 ライブインスピレーション配信 🐱",
        "select_mode": "1. モードの選択",
        "prompt": "プロンプト",
        "num_images": "生成する画像数",
        "upload_sketch": "スケッチをアップロードまたは描画",
        "generate_btn": "バッチ生成",
        "preview_label": "処理されたエッジマップのプレビュー",
        "gallery_label": "生成された出力画像",
        "err_fields": "### ⚠️ すべての項目が必須です！",
        "err_match": "### ❌ パスワードが一致しません",
        "err_user_taken": "### ❌ このユーザー名はすでに使用されています。",
        "err_email_taken": "### ❌ このメールアドレスはすでに登録されています。",
        "success_reg": "### 🎉 ようこそ！\nスタジオにリダイレクト中..."
    },
    "Arabic (العربية)": {
        "welcome_title": "# 🚀 مرحبًا بكم في منصة الابتكار الخاصة بنا",
        "welcome_subtitle": "### اختبر قوة أدوات الجيل القادم من الذكاء الاصطناعي المصممة خصيصًا لك.",
        "why_join_title": "## لماذا تنضم إلينا؟",
        "why_join_text": "* **وصول فوري:** قم ببناء واختبار ونشر النماذج باستخدام خطوط معالجة رائدة.\n* **قوة السحاب:** يتم التعامل مع الأحمال الحسابية الثقيلة بسلاسة.",
        "features_preview": "### معاينة الميزات",
        "create_account": "## أنشئ حسابك المجاني",
        "username": "اسم المستخدم",
        "email": "البريد الإلكتروني",
        "password": "كلمة المرور",
        "confirm_password": "تأكيد كلمة المرور",
        "register_btn": "سجل الآن",
        "show_btn": "👁️ عرض",
        "hide_btn": "🙈 إخفاء",
        "studio_title": "# 🎨 استوديو توليد الصور بالذكاء الاصطناعي (وضع الدفعات)",
        "inspo_stream": "### 🐶 بث الإلهام المباشر 🐱",
        "select_mode": "1. اختر الوضع",
        "prompt": "المطالبة (Prompt)",
        "num_images": "عدد الصور المراد توليدها",
        "upload_sketch": "تحميل أو رسم مسودة",
        "generate_btn": "توليد الدفعة",
        "preview_label": "معاينة خريطة الحواف المعالجة",
        "gallery_label": "الصور الناتجة المُولدة",
        "err_fields": "### ⚠️ جميع الحقول مطلوبة!",
        "err_match": "### ❌ كلمات المرور غير متطابقة",
        "err_user_taken": "### ❌ اسم المستخدم مأخوذ بالفعل.",
        "err_email_taken": "### ❌ هذا البريد الإلكتروني مسجل بالفعل.",
        "success_reg": "### 🎉 مرحبًا بك معنا!\nجاري التوجيه إلى الاستوديو..."
    }
}

# --- 5. IMAGE SLIDER DATA & LOGIC ---
SLIDER_IMAGES = [
    "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=500",
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=500",
    "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?w=500",
    "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=500",
]
images_list = [f"https://picsum.photos/300/300?random={i}" for i in range(1, 21)]

def rotate_slider(current_index):
    next_index = (current_index + 1) % len(SLIDER_IMAGES)
    return SLIDER_IMAGES[next_index], next_index

def preprocess_sketch(pil_image):
    if pil_image is None:
        return None
    img = pil_image.convert('RGB')
    gray_img = img.convert('L')
    np_img = np.array(gray_img)
    edges = cv2.Canny(np_img, 100, 200)
    final_np_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_np_img).resize((512, 512))

# --- 6. GRADIO ACTIONS & ENGINE WORKFLOWS ---
def save_user(username, email, password, confirm_password, lang_choice):
    """Validates properties and registers user using dynamic runtime language settings."""
    username = username.strip()
    email = email.strip()
    trans = LANGUAGES[lang_choice]
    
    if not username or not email or not password or not confirm_password:
        return trans["err_fields"], gr.update(), gr.update()
        
    if password != confirm_password:
        return trans["err_match"], gr.update(), gr.update()
    
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            
            cursor.execute("SELECT Username FROM users WHERE Username = %s", (username,))
            if cursor.fetchone():
                return trans["err_user_taken"], gr.update(), gr.update()
                
            cursor.execute("SELECT Email FROM users WHERE Email = %s", (email,))
            if cursor.fetchone():
                return trans["err_email_taken"], gr.update(), gr.update()
            
            secure_password_string = hash_password(password)
            insert_query = "INSERT INTO users (Username, Email, Password) VALUES (%s, %s, %s)"
            cursor.execute(insert_query, (username, email, secure_password_string))
            connection.commit()
            
            return trans["success_reg"], gr.update(visible=False), gr.update(visible=True)
            
    except Error as e:
        return f"### ❌ Database Error\n{e}", gr.update(), gr.update()
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def change_language(lang_choice):
    """Dynamically rewrites every component label and layout text on configuration shift."""
    t = LANGUAGES[lang_choice]
    return [
        gr.update(value=t["welcome_title"]),
        gr.update(value=t["welcome_subtitle"]),
        gr.update(value=t["why_join_title"]),
        gr.update(value=t["why_join_text"]),
        gr.update(value=t["features_preview"]),
        gr.update(value=t["create_account"]),
        gr.update(label=t["username"]),
        gr.update(label=t["email"]),
        gr.update(label=t["password"]),
        gr.update(value=t["show_btn"]),
        gr.update(label=t["confirm_password"]),
        gr.update(value=t["show_btn"]),
        gr.update(value=t["register_btn"]),
        gr.update(value=t["studio_title"]),
        gr.update(value=t["inspo_stream"]),
        gr.update(label=t["select_mode"]),
        gr.update(label=t["prompt"]),
        gr.update(label=t["num_images"]),
        gr.update(label=t["upload_sketch"]),
        gr.update(value=t["generate_btn"]),
        gr.update(label=t["preview_label"]),
        gr.update(label=t["gallery_label"])
    ]

def generate(mode, sketch_img, prompt, num_images, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Please enter a style/content prompt!")

    batch_size = int(num_images)
    processed_sketch = None
    prompts = [prompt] * batch_size
    progress(0, desc="Waiting in queue / Initializing...")

    if mode == "Sketch to Image":
        if sketch_img is None:
            raise gr.Error("Please upload or draw a sketch first!")
        processed_sketch = preprocess_sketch(sketch_img)

    with gpu_lock:
        progress(0.2, desc="Processing batch on GPU...")
        if mode == "Sketch to Image":
            output_images = pipe_sketch2img(
                prompt=prompts,
                image=[processed_sketch] * batch_size,
                controlnet_conditioning_scale=1.0,
                guidance_scale=7.5,
                num_inference_steps=20
            ).images
        else:
            output_images = pipe_text2img(
                prompt=prompts,
                guidance_scale=7.5,
                num_inference_steps=20
            ).images

    saved_filenames = []
    for image in output_images:
        unique_filename = f"generation_{uuid.uuid4().hex}.png"
        save_path = os.path.join(OUTPUT_DIR, unique_filename)
        image.save(save_path)
        saved_filenames.append(unique_filename)

    with log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filenames_str = "; ".join(saved_filenames)
        with open(CSV_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, mode, prompt, batch_size, filenames_str])

    preview = processed_sketch if mode == "Sketch to Image" else gr.update(visible=False)
    return preview, output_images

# --- 7. PREMIUM LUXURY CSS STYLING ---
custom_css = """
.gradio-container {
    background: transparent !important;
    min-height: 100vh;
    padding: 40px 30px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
#dynamic-bg-canvas {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    background: #09090e;
}
.gradio-container h1, .gradio-container h2, .gradio-container h3, 
.gradio-container p, .gradio-container span, .gradio-container label, .gradio-container .prose {
    color: #ffffff !important;
}
.top-header-row {
    display: flex; justify-content: space-between; align-items: center; padding: 10px 0 30px 0;
}
.top-nav {
    display: flex; gap: 40px; font-size: 13px; font-weight: 700;
    letter-spacing: 2px; color: rgba(255, 255, 255, 0.6); align-items: center;
}
.top-nav span { cursor: pointer; transition: all 0.3s ease; }
.top-nav span:hover { color: #ffffff !important; text-shadow: 0 0 15px rgba(255, 255, 255, 0.6); }
.lang-selector { width: 200px !important; }
.glass-panel {
    background: rgba(255,
