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
except ImportError as e:
    st.error(f"Error importing modules: {e}. Make sure read.py, model.py, and utils.py are in the same directory.")
    st.stop()

# --- Configuration ---
st.set_page_config(
    page_title="Urdu OCR - UTRNet",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Setup ---
BASE_DIR = os.path.dirname(current_dir) # dashboard/
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Projects/ML/

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

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
    
    # 2. Load Recognition Model
    recognition_model = Model(num_class=len(converter.character), device=device)
    recognition_model = recognition_model.to(device)
    
    rec_weights_path = os.path.join(MODELS_DIR, "best_norm_ED.pth")
    if os.path.exists(rec_weights_path):
        recognition_model.load_state_dict(torch.load(rec_weights_path, map_location=device))
        recognition_model.eval()
    else:
        st.warning(f"Recognition weights not found at {rec_weights_path}")
    
    # 3. Load Detection Model
    det_weights_path = os.path.join(MODELS_DIR, "yolov8m_UrduDoc.pt")
    if os.path.exists(det_weights_path):
        detection_model = YOLO(det_weights_path)
    else:
        st.error(f"YOLO weights not found at {det_weights_path}")
        detection_model = None

    return recognition_model, detection_model, converter, device

# Initialize resources
rec_model, det_model, converter, device = load_resources()

if not (rec_model and det_model):
    st.stop()

# --- Helper Functions ---
def format_fir(lines):
    """Lightweight FIR formatter."""
    sections = []
    for idx, text in enumerate(lines, start=1):
        sections.append({"section": f"Line {idx}", "text": text})
    return {"status": "processed", "count": len(lines), "lines": sections}

def run_pipeline(image):
    """Run the Full OCR Pipeline."""
    status_log = []
    
    # Status Container
    status_container = st.status("🚀 Starting OCR Pipeline...", expanded=True)
    
    # 1. Detection
    status_container.write("🔍 Stage 1: Detecting text lines...")
    start_time = time.time()
    
    results = det_model.predict(
        source=image,
        conf=0.01,
        imgsz=1280,
        save=False,
        agnostic_nms=True,
        max_det=300,
        device=device,
        verbose=False
    )
    
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
    boxes.sort(key=lambda x: x[1]) # Sort by Y coordinate
    
    elapsed = time.time() - start_time
    status_container.write(f"   ✓ Found {len(boxes)} lines in {elapsed:.2f}s")
    
    # 2. Visualization
    status_container.write("🎨 Stage 2: Creating visualization...")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    
    for i, box in enumerate(boxes):
        color = tuple(np.random.randint(50, 255, 3))
        draw.rectangle(box, outline=color, width=4)
        draw.text((box[0] + 5, box[1] + 5), str(i + 1), fill=color)
    
    # 3. Cropping
    status_container.write("✂️ Stage 3: Cropping text lines...")
    crops = [image.crop(box) for box in boxes]
    
    # 4. OCR
    status_container.write("📖 Stage 4: Running OCR on crops...")
    ocr_texts = []
    ocr_start = time.time()
    
    prog_bar = status_container.progress(0)
    
    for idx, crop in enumerate(crops):
        try:
            text = text_recognizer(crop, rec_model, converter, device)
            ocr_texts.append(text)
        except Exception as e:
            ocr_texts.append(f"[Error: {e}]")
        
        # Update progress
        prog_bar.progress((idx + 1) / len(crops))
    
    ocr_elapsed = time.time() - ocr_start
    status_container.write(f"   ✓ OCR completed in {ocr_elapsed:.2f}s")
    
    # 5. Finalize
    status_container.update(label="✅ Pipeline Completed!", state="complete", expanded=False)
    
    return overlay, crops, ocr_texts, format_fir(ocr_texts)


# --- UI Layout ---
st.title("🌙 Urdu OCR - UTRNet Dashboard")
st.markdown("Upload a document or FIR image to extract Urdu text using the UTRNet pipeline.")

# Sidebar
with st.sidebar:
    st.header("Input Source")
    input_method = st.radio("Choose input:", ["Upload Image", "Use Example"])
    
    image_file = None
    if input_method == "Upload Image":
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    else:
        # Load examples
        images_dir = os.path.join(STATIC_DIR, "images")
        if os.path.exists(images_dir):
            example_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
            selected_example = st.selectbox("Select an example:", example_files)
            if selected_example:
                image_file = os.path.join(images_dir, selected_example)
        else:
            st.warning("No examples found in static/images")

    st.divider()
    st.info(f"Device: {device}")
    st.info(f"GPU Available: {torch.cuda.is_available()}")

# Main Logic
if image_file:
    # Load Image
    try:
        if isinstance(image_file, str):
            image = Image.open(image_file).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
            
        st.image(image, caption="Input Image", use_container_width=True)
        
        if st.button("▶️ Run OCR Extraction", type="primary"):
            
            overlay_img, cropped_imgs, texts, json_output = run_pipeline(image)
            
            # --- Results Display ---
            st.divider()
            st.header("📝 Results")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📄 Full Text", "🔍 Detection Overlay", "✂️ Line Details", "💾 JSON Data"])
            
            with tab1:
                full_text = "\n".join(texts)
                st.text_area("Extracted Text", value=full_text, height=400)
                st.download_button("Download Text", full_text, "urdu_ocr_result.txt")
            
            with tab2:
                st.image(overlay_img, caption="Detected Text Lines", use_container_width=True)
                
            with tab3:
                st.write(f"Extracted {len(cropped_imgs)} lines:")
                # Display lines in a grid or list
                for i, (crop, txt) in enumerate(zip(cropped_imgs, texts)):
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.image(crop, use_container_width=True)
                    with c2:
                        st.markdown(f"**Line {i+1}:**")
                        st.text_input(label=f"hidden_l{i}", value=txt, label_visibility="collapsed", key=f"txt_{i}")
                    st.divider()
            
            with tab4:
                st.json(json_output)
                
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("👈 Please upload an image or select an example to begin.")