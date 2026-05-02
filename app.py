import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
import os

st.set_page_config(
    page_title="ImageNet Classifier",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Theme Toggle ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

col_spacer_l, col_toggle, col_spacer_r = st.columns([5, 2, 1])
with col_toggle:
    theme_label = "🌙 Dark" if st.session_state.dark_mode else "☀️ Light"
    if st.button(theme_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

dark = st.session_state.dark_mode

# --- Theme Variables ---
if dark:
    bg_primary = "#0f1923"
    bg_card = "rgba(255,255,255,0.06)"
    bg_card_hover = "rgba(255,255,255,0.1)"
    text_primary = "#ffffff"
    text_secondary = "rgba(255,255,255,0.6)"
    text_muted = "rgba(255,255,255,0.4)"
    border_color = "rgba(255,255,255,0.1)"
    accent_main = "#00f2fe"
    accent_alt = "#ff6b6b"
    gradient_title = "linear-gradient(135deg, #ffffff, #b3e5fc)"
    gradient_name = "linear-gradient(135deg, #00f2fe, #4facfe, #00f2fe)"
    divider_glow = "linear-gradient(90deg, transparent, rgba(0,242,254,0.35), transparent)"
    shadow = "0 8px 30px rgba(0,0,0,0.3)"
    stapp_bg = "linear-gradient(-45deg, #0f2027, #203a43, #2c5364)"
    label_color = "#ffffff"
else:
    bg_primary = "#f5f7fa"
    bg_card = "rgba(0,0,0,0.03)"
    bg_card_hover = "rgba(0,0,0,0.06)"
    text_primary = "#1a1a2e"
    text_secondary = "rgba(0,0,0,0.55)"
    text_muted = "rgba(0,0,0,0.35)"
    border_color = "rgba(0,0,0,0.08)"
    accent_main = "#0097a7"
    accent_alt = "#e53935"
    gradient_title = "linear-gradient(135deg, #1a1a2e, #2c3e50)"
    gradient_name = "linear-gradient(135deg, #0097a7, #00bcd4, #0097a7)"
    divider_glow = "linear-gradient(90deg, transparent, rgba(0,151,167,0.3), transparent)"
    shadow = "0 8px 30px rgba(0,0,0,0.08)"
    stapp_bg = "linear-gradient(135deg, #f5f7fa, #e4e9f0, #f5f7fa)"
    label_color = "#1a1a2e"

# --- CSS ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    .stApp {{
        background: {stapp_bg} !important;
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }}
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    .stApp label, .stApp .stFileUploader label,
    .stApp [data-testid="stFileUploader"] label,
    .stApp [data-testid="stExpander"] summary span,
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stMarkdownContainer"] h4,
    .stApp .stSpinner > div,
    .stApp [data-testid="stImage"] div {{
        color: {label_color} !important;
    }}

    .hero-title {{
        font-size: 52px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        font-family: 'Outfit', sans-serif;
        background: {gradient_title};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 0.8s ease-out;
    }}
    .hero-subtitle {{
        font-size: 18px;
        color: {text_secondary};
        text-align: center;
        margin-bottom: 35px;
        font-weight: 300;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.5px;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .result-card {{
        background: {bg_card};
        border-radius: 16px;
        padding: 28px 20px;
        margin-top: 20px;
        text-align: center;
        border: 1px solid {border_color};
        box-shadow: {shadow};
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .result-card:hover {{
        transform: translateY(-5px);
        background: {bg_card_hover};
    }}
    .result-card.prediction-card {{ border-left: 6px solid {accent_main}; }}

    .result-emoji {{ font-size: 48px; display: block; margin-bottom: 10px; }}
    .result-label {{
        font-size: 26px;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        margin: 0;
    }}
    .result-label.main {{ color: {accent_main}; }}
    .confidence-bar {{
        background: linear-gradient(90deg, {accent_main}, {accent_alt});
        height: 8px;
        border-radius: 4px;
        margin-top: 12px;
        overflow: hidden;
    }}
    .confidence-bar-fill {{
        height: 100%;
        background: {accent_main};
        transition: width 0.3s ease;
    }}

    .result-confidence {{
        font-size: 15px;
        color: {text_secondary};
        margin-top: 8px;
        font-family: 'Outfit', sans-serif;
    }}

    .glow-divider {{
        height: 1px;
        background: {divider_glow};
        margin: 50px 0 25px 0;
        border: none;
    }}

    .footer-name {{
        font-size: 22px;
        font-weight: 800;
        text-align: center;
        letter-spacing: 2px;
        font-family: 'Outfit', sans-serif;
        background: {gradient_name};
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
    }}
    @keyframes shimmer {{
        to {{ background-position: 200% center; }}
    }}
    </style>
""", unsafe_allow_html=True)


# --- Emoji mapping by ImageNet class index ranges & keywords ---
def get_emoji_for_label(label, idx):
    """Return a contextual emoji based on the predicted class."""
    label_lower = label.lower()

    # Dogs (indices 152-268 in ImageNet)
    if 152 <= idx <= 268:
        return "🐕"
    # Cats (indices 281-285)
    if 281 <= idx <= 285:
        return "🐱"
    # Big cats
    if idx in range(286, 295):
        return "🐅"
    # Birds
    if (7 <= idx <= 24) or (80 <= idx <= 100) or (128 <= idx <= 147):
        return "🐦"
    # Fish
    if 389 <= idx <= 397:
        return "🐟"
    # Reptiles / snakes
    if 33 <= idx <= 68:
        return "🐍"
    # Insects / spiders
    if 69 <= idx <= 79 or 300 <= idx <= 327:
        return "🐛"
    # Bears
    if 295 <= idx <= 298:
        return "🐻"
    # Primates
    if 365 <= idx <= 384:
        return "🐵"
    # Marine
    if 108 <= idx <= 127 or 148 <= idx <= 151:
        return "🐋"
    # Vehicles / transport
    if "car" in label_lower or "truck" in label_lower or "bus" in label_lower:
        return "🚗"
    if "boat" in label_lower or "ship" in label_lower:
        return "🚢"
    if "plane" in label_lower or "airliner" in label_lower or "aircraft" in label_lower:
        return "✈️"
    # Food
    if 924 <= idx <= 968:
        return "🍽️"
    # Flowers / plants
    if 985 <= idx <= 988:
        return "🌸"
    # Furniture
    if "chair" in label_lower or "table" in label_lower or "desk" in label_lower:
        return "🪑"
    # Musical instruments
    if any(w in label_lower for w in ["guitar", "piano", "violin", "drum", "flute", "organ", "harp", "cello", "banjo"]):
        return "🎵"
    # Electronics
    if any(w in label_lower for w in ["computer", "laptop", "monitor", "keyboard", "phone", "television"]):
        return "💻"
    # Default
    return "✨"


@st.cache_resource
def load_model():
    """Load the ONNX model and ImageNet labels."""
    model_path = "mobilenetv2.onnx"
    labels_path = "imagenet_classes.txt"
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None, None
    session = ort.InferenceSession(model_path)
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return session, labels


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)


def predict_image(image, session, labels):
    """Preprocess the image and get predictions using ONNX Runtime."""
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array.astype(np.float32)})[0]
    probs = softmax(outputs)[0]
    top_indices = np.argsort(probs)[-10:][::-1]

    top_pred_idx = top_indices[0]
    top_pred_label = labels[top_pred_idx]
    top_pred_conf = probs[top_pred_idx] * 100

    raw_preds = [{"label": labels[idx], "score": float(probs[idx]), "idx": int(idx)} for idx in top_indices]
    return top_pred_label, top_pred_conf, top_pred_idx, raw_preds


# ===================== UI =====================
st.markdown("<div class='hero-title'>🖼️ ImageNet Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Upload any image and let AI identify it from 1,000 ImageNet classes!</div>", unsafe_allow_html=True)

session, labels = load_model()

if session is None:
    st.error("⚠️ Model files not found! Please ensure 'mobilenetv2.onnx' and 'imagenet_classes.txt' are in the directory.")
else:
    uploaded_file = st.file_uploader("Choose an image (JPG / PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_container_width=True)
        with col2:
            st.markdown("#### 🔍 Top Prediction")
            with st.spinner("Analyzing image..."):
                pred_label, confidence, pred_idx, raw_preds = predict_image(image, session, labels)
                emoji = get_emoji_for_label(pred_label, pred_idx)
                st.markdown(f"""
                    <div class='result-card prediction-card'>
                        <span class='result-emoji'>{emoji}</span>
                        <p class='result-label main'>{pred_label.title()}</p>
                        <p class='result-confidence'>Confidence: {confidence:.2f}%</p>
                        <div class='confidence-bar'>
                            <div class='confidence-bar-fill' style='width: {confidence}%'></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with st.expander("📊 Show Top 10 Predictions"):
                for i, pred in enumerate(raw_preds):
                    e = get_emoji_for_label(pred['label'], pred['idx'])
                    st.write(f"{i+1}. {e} **{pred['label'].title()}** — {pred['score']*100:.2f}%")

# ===================== Footer =====================
st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='footer-name'>✨ Created by Aryan Yadav ✨</div>", unsafe_allow_html=True)
