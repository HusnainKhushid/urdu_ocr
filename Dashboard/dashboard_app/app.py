import os
import sys
import time
import torch
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO

# Add current directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from read import text_recognizer
    from model import Model
    from utils import CTCLabelConverter
    from gemini_extractor import extract_fields_from_image, format_fields_for_display
except ImportError as e:
    st.error(f"Error importing modules: {e}. Make sure read.py, model.py, and utils.py are in the same directory.")
    st.stop()

# --- Configuration ---
st.set_page_config(
    page_title="Urdu OCR - UTRNet",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Setup ---
BASE_DIR = os.path.dirname(current_dir) # dashboard/
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Projects/ML/
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# --- Session State Initialization ---
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'input_image' not in st.session_state:
    st.session_state.input_image = None
if 'uploaded_file_id' not in st.session_state:
    st.session_state.uploaded_file_id = None

# --- Resource Loading ---
@st.cache_resource
def load_resources():
    """Load models and converters once."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Vocabulary
    vocab_path = os.path.join(DATA_DIR, "UrduGlyphs.txt")
    if not os.path.exists(vocab_path):
        st.error(f"Vocabulary file not found at: {vocab_path}")
        return None, None, None, None
    with open(vocab_path, "r", encoding="utf-8") as f:
        content = f.readlines()
    content = ''.join([str(elem).strip('\n') for elem in content])
    content = content + " "
    converter = CTCLabelConverter(content)
    
    # 2. Recognition Model
    recognition_model = Model(num_class=len(converter.character), device=device)
    recognition_model = recognition_model.to(device)
    rec_weights = os.path.join(MODELS_DIR, "best_norm_ED.pth")
    if os.path.exists(rec_weights):
        recognition_model.load_state_dict(torch.load(rec_weights, map_location=device))
        recognition_model.eval()
    
    # 3. Detection Model
    det_weights = os.path.join(MODELS_DIR, "yolov8m_UrduDoc.pt")
    detection_model = YOLO(det_weights) if os.path.exists(det_weights) else None

    return recognition_model, detection_model, converter, device

rec_model, det_model, converter, device = load_resources()
if not (rec_model and det_model):
    st.stop()

# --- Logic ---
def run_pipeline(image):
    """Run OCR and store results in session state."""
    # Simplified status: handled by caller with spinner
    
    start = time.time()
    
    # 1. Detection
    # status_container.write("🔍 Detecting lines...")
    results = det_model.predict(image, conf=0.01, imgsz=1280, agnostic_nms=True, max_det=300, verbose=False, device=device)
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
    boxes.sort(key=lambda x: x[1])
    # status_container.write(f"   ✓ Found {len(boxes)} lines")
    
    # 2. Vis & Crop
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    crops = []
    for i, box in enumerate(boxes):
        color = tuple(np.random.randint(50, 255, 3))
        draw.rectangle(box, outline=color, width=4)
        draw.text((box[0]+5, box[1]+5), str(i+1), fill=(0, 0, 0))  # Black text
        crops.append(image.crop(box))
    
    # 3. OCR
    # status_container.write("📖 Reading text...")
    ocr_texts = []
    # prog = status_container.progress(0)
    for idx, crop in enumerate(crops):
        try:
            txt = text_recognizer(crop, rec_model, converter, device)
            ocr_texts.append(txt)
        except:
            ocr_texts.append("")
        # prog.progress((idx+1)/len(crops))
    
    # 4. Gemini Field Extraction (prints to terminal)
    print("\n" + "=" * 60)
    print("📤 Sending image to Gemini for field extraction...")
    gemini_data = extract_fields_from_image(image, print_to_terminal=True)
    gemini_formatted = format_fields_for_display(gemini_data)
    
    # Also print OCR results to terminal
    print("\n" + "=" * 60)
    print("📖 UTRNet OCR RESULTS")
    print("=" * 60)
    for i, txt in enumerate(ocr_texts):
        print(f"  Line {i+1}: {txt}")
    print("=" * 60 + "\n")
    
    # status_container.update(label="✅ Done!", state="complete", expanded=False)
    
    # Store results
    st.session_state.results = {
        "text": "\n".join(ocr_texts),
        "overlay": overlay,
        "crops": crops,
        "line_texts": ocr_texts,
        "json": [{"line": i+1, "text": t} for i, t in enumerate(ocr_texts)],
        "gemini_data": gemini_data,
        "gemini_text": gemini_formatted
    }
    st.session_state.processed = True

# --- Custom CSS ---
def inject_custom_css():
    st.markdown("""
        <style>
        /* Global Font */
        html, body, [class*="css"] {
            font-family: 'Helvetica', sans-serif;
            font-weight: bold;
            color: #1a1a1a;
        }
        
        /* App Background */
        .stApp {
            background-color: #f3fcf4; /* Very light slightly green tint */
        }
        
        /* Box/Card Styling - Official Document Look */
        .custom-card {
            background-color: white;
            padding: 30px;
            border-radius: 4px; /* Sharper corners */
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05); /* Softer shadow */
            height: 100%;
            border: 1px solid #e2e8f0;
            border-top: 5px solid #115740; /* Pakistan Green Top Border */
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        /* Force dark text color for all markdown content */
        .custom-card p, .custom-card span, .custom-card div {
            color: #1a1a1a !important;
        }
        
        /* Markdown text styling */
        .stMarkdown, .stMarkdown p, .stMarkdown span {
            color: #1a1a1a !important;
        }
        
        /* Ensure all text elements are visible */
        [data-testid="stMarkdownContainer"] p {
            color: #1a1a1a !important;
        }

        /* Headers */
        h1 {
            color: #115740; /* Pakistan Green */
            font-family: 'Helvetica', sans-serif;
            font-weight: 800; /* Extra Bold */
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }
        h2, h3 {
            color: #115740;
            font-family: 'Helvetica', sans-serif;
            font-weight: bold;
            text-align: center;
        }
        h4, h5 {
            color: #856404; /* Gold/Dark Yellowish */
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 0;
        }
        
        /* Buttons - Official Green */
        .stButton {
            display: flex;
            justify-content: center;
            width: 100%;
        }
        .stButton>button {
            width: 80% !important;
            border-radius: 4px;
            background-color: #115740;
            color: white;
            border: none;
            font-weight: bold;
            padding-top: 0.7rem;
            padding-bottom: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: background 0.3s;
        }
        .stButton>button:hover {
            background-color: #0d4231;
            color: #FFD700; /* Gold text on hover */
        }
        
        /* File Uploader */
        .stFileUploader {
            padding: 1.5rem;
            border: 2px dashed #115740; /* Green dashed border */
            border-radius: 4px;
            background-color: #ffffff;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        /* Sidebar - Official Dark */
        section[data-testid="stSidebar"] {
            background-color: #06241b; /* Very Dark Green */
            color: white;
            padding-top: 1rem;
            border-right: 5px solid #bd9b60; /* Gold border */
        }
        
        section[data-testid="stSidebar"] h1 {
            color: white !important;
            text-align: left !important;
            font-size: 1.2rem !important;
            padding-left: 0.5rem;
        }
        
        div[data-testid="stSidebarNav"] li div a {
             color: #ffffff;
             font-weight: normal;
        }
        div[data-testid="stSidebarNav"] li div a:hover {
             color: #FFD700;
        }

        /* Images */
        div[data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }
        div[data-testid="stImage"] > img {
            border: 1px solid #ddd;
            padding: 5px;
            background: white;
        }

        </style>
    """, unsafe_allow_html=True)

# --- UI Components ---
def render_header():
    """Renders the official Government of Pakistan header."""
    logo_path = os.path.join(STATIC_DIR, "images", "pak_logo.png")
    
    col_l, col_m, col_r = st.columns([1, 6, 1])
    with col_l:
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
    with col_m:
        st.markdown("""
            <div style='text-align: center;'>
                <h4 style='margin-bottom: 0; color: #115740;'>GOVERNMENT OF PAKISTAN</h4>
                <h3 style='margin-top: 0; margin-bottom: 5px; font-size: 1.2rem; color: #1a1a1a;'>MINISTRY OF INFORMATION TECHNOLOGY & TELECOMMUNICATION</h3>
                <h1 style='font-size: 2.5rem; margin-top: 10px;'>OFFICIAL DOCUMENT DIGITIZATION PORTAL</h1>
            </div>
        """, unsafe_allow_html=True)
    with col_r:
         if os.path.exists(logo_path):
            st.image(logo_path, width=100)
            
    st.markdown("<hr style='border-top: 3px solid #bd9b60; margin-top: 0;'>", unsafe_allow_html=True)

def render_ocr_view():
    inject_custom_css()
    render_header()
    
    # Main Layout: Two Columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    # Left Column: Input
    with col1:
        st.markdown('<div class="custom-card"><h3>INPUT DOCUMENT</h3>', unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.9rem; color: #666;'>Upload official correspondence or gazettes (JPEG/PNG)</p>", unsafe_allow_html=True)
        
        # Upload Logic
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="uploader_ocr", label_visibility="collapsed")
        
        if uploaded_file:
            # Check if new file
            if uploaded_file.file_id != st.session_state.uploaded_file_id:
                image = Image.open(uploaded_file).convert("RGB")
                st.session_state.input_image = image
                st.session_state.uploaded_file_id = uploaded_file.file_id
                st.session_state.processed = False # Reset processing
        
        if st.session_state.input_image:
            st.image(st.session_state.input_image, caption="Document Preview", use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("PROCESS DOCUMENT", type="primary"):
                with st.spinner("Authenticating and extracting text..."):
                    run_pipeline(st.session_state.input_image)
                st.rerun()
        else:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.info("System Ready. Awaiting Input.")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Right Column: Output
    with col2:
        st.markdown('<div class="custom-card"><h3>DIGITIZATION RESULTS</h3>', unsafe_allow_html=True)
        
        if st.session_state.processed:
            st.success("Extraction Successful")
            
            # Display Gemini Extracted Fields (primary output)
            st.markdown("<p style='font-size: 0.9rem; color: #115740; font-weight: bold;'>📋 EXTRACTED FIELDS</p>", unsafe_allow_html=True)
            gemini_data = st.session_state.results.get('gemini_data', {})
            
            if gemini_data and not gemini_data.get('error'):
                # Display main fields in a clean format
                if gemini_data.get('serial_number'):
                    st.markdown(f"**📌 Serial Number:** {gemini_data['serial_number']}")
                if gemini_data.get('name_urdu') or gemini_data.get('name_english'):
                    st.markdown(f"**👤 Name:** {gemini_data.get('name_urdu', '')}  —  {gemini_data.get('name_english', '')}")
                if gemini_data.get('father_name_urdu') or gemini_data.get('father_name_english'):
                    st.markdown(f"**👨 Father:** {gemini_data.get('father_name_urdu', '')}  —  {gemini_data.get('father_name_english', '')}")
                if gemini_data.get('cnic'):
                    st.markdown(f"**🪪 CNIC:** {gemini_data['cnic']}")
                if gemini_data.get('date'):
                    st.markdown(f"**📅 Date:** {gemini_data['date']}")
                if gemini_data.get('address_urdu') or gemini_data.get('address_english'):
                    st.markdown(f"**📍 Address:** {gemini_data.get('address_urdu', '')}  —  {gemini_data.get('address_english', '')}")
                
                # Display numbered fields
                if gemini_data.get('fields'):
                    st.markdown("---")
                    st.markdown("**Numbered Fields:**")
                    for field in gemini_data['fields']:
                        num = field.get('number', '?')
                        label = field.get('label', 'Field')
                        val_u = field.get('value_urdu', '-')
                        val_e = field.get('value_english', '-')
                        st.markdown(f"**[{num}] {label}:** {val_u}  —  {val_e}")
            else:
                gemini_text = st.session_state.results.get('gemini_text', 'No fields extracted')
                st.info(gemini_text)
            
            # Collapsible OCR raw text
            with st.expander("📖 Raw OCR Text"):
                result_text = st.session_state.results.get('text', '')
                st.text_area("OCR Output", value=result_text, height=150, label_visibility="collapsed")
            
            # Download actions
            st.download_button(
                label="DOWNLOAD TRANSCRIPT",
                data=st.session_state.results.get('text', ''),
                file_name="official_transcript.txt",
                mime="text/plain",
                type="secondary"
            )
        else:
            # Empty state
            st.markdown("""
                <div style='text-align: center; color: #6b7280; padding: 4rem 0;'>
                    <p style='font-weight: bold;'>NO DATA GENERATED</p>
                    <p style='font-size: 0.8rem;'>Upload a document to generate an official record.</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

def render_pipeline_view():
    st.header("Pipeline Overview")
    st.markdown("Detailed inspection of the OCR process.")
    
    if not st.session_state.processed:
        if st.session_state.input_image:
            st.warning("Image loaded but not processed. Go to 'OCR' tab or click Run below.")
            if st.button("Run Pipeline Debug"):
                run_pipeline(st.session_state.input_image)
                st.rerun()
        else:
            st.info("Please upload an image in the 'OCR' section first.")
        return

    # Tabs for details
    t1, t2, t3, t4 = st.tabs(["Detection", "Line Analysis", "Gemini Analysis", "Raw Data"])
    
    with t1:
        st.image(st.session_state.results['overlay'], caption="YOLOv8 Line Detection", use_container_width=True)
    
    with t2:
        crops = st.session_state.results['crops']
        texts = st.session_state.results['line_texts']
        st.write(f"Total Lines: {len(crops)}")
        for i, (cr, tx) in enumerate(zip(crops, texts)):
            c1, c2 = st.columns([1, 4])
            with c1: st.image(cr)
            with c2: st.code(tx, language=None)
            st.divider()
    
    with t3:
        st.markdown("### 🤖 Gemini 2.5 Flash - Extracted Fields")
        gemini_data = st.session_state.results.get('gemini_data', {})
        
        if gemini_data and not gemini_data.get('error'):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Main Information**")
                if gemini_data.get('serial_number'):
                    st.info(f"📌 Serial: {gemini_data['serial_number']}")
                if gemini_data.get('name_urdu') or gemini_data.get('name_english'):
                    st.success(f"👤 {gemini_data.get('name_urdu', '')} | {gemini_data.get('name_english', '')}")
                if gemini_data.get('cnic'):
                    st.info(f"🪪 CNIC: {gemini_data['cnic']}")
                if gemini_data.get('date'):
                    st.info(f"📅 Date: {gemini_data['date']}")
            
            with col_b:
                st.markdown("**Family & Address**")
                if gemini_data.get('father_name_urdu') or gemini_data.get('father_name_english'):
                    st.info(f"👨 Father: {gemini_data.get('father_name_urdu', '')} | {gemini_data.get('father_name_english', '')}")
                if gemini_data.get('address_urdu') or gemini_data.get('address_english'):
                    st.info(f"📍 {gemini_data.get('address_urdu', '')} | {gemini_data.get('address_english', '')}")
            
            # Numbered fields as expandable
            if gemini_data.get('fields'):
                st.markdown("---")
                st.markdown("**All Numbered Fields**")
                for field in gemini_data['fields']:
                    with st.expander(f"Field [{field.get('number', '?')}] - {field.get('label', 'Field')}"):
                        st.write(f"**Urdu:** {field.get('value_urdu', '-')}")
                        st.write(f"**English:** {field.get('value_english', '-')}")
            
            # Show raw JSON
            with st.expander("📦 Raw JSON Response"):
                st.json(gemini_data)
        else:
            st.warning(st.session_state.results.get('gemini_text', 'No Gemini analysis available'))
            
    with t4:
        st.json(st.session_state.results['json'])


# --- Main Layout ---
st.sidebar.title("Navigation")
view = st.sidebar.radio("Go to:", ["OCR", "Pipeline Overview"])

st.sidebar.divider()
st.sidebar.info("Use 'OCR' for quick results and 'Pipeline Overview' for debugging.")

if view == "OCR":
    render_ocr_view()
else:
    render_pipeline_view()