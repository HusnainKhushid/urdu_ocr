import os
import json
import torch
import gradio as gr
from ultralytics import YOLO
from PIL import ImageDraw

from .read import text_recognizer
from .model import Model
from .utils import CTCLabelConverter

from .gemini_extractor import extract_fields_from_image, format_fields_for_display


# Add the parent directory to the Python path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

""" vocab / character number configuration """
file = open(os.path.join(PROJECT_ROOT, "data", "UrduGlyphs.txt"),"r",encoding="utf-8")
content = file.readlines()
content = ''.join([str(elem).strip('\n') for elem in content])
content = content+" "
""" model configuration """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
converter = CTCLabelConverter(content)
recognition_model = Model(num_class=len(converter.character), device=device)
recognition_model = recognition_model.to(device)
recognition_model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "models", "best_norm_ED.pth"), map_location=device))
recognition_model.eval()

detection_model = YOLO(os.path.join(PROJECT_ROOT, "models", "yolov8m_UrduDoc.pt"))

examples = [os.path.join(BASE_DIR, "static", "images", name) for name in ["1.jpg", "2.jpg", "3.jpg"] if os.path.exists(os.path.join(BASE_DIR, "static", "images", name))]

def format_fir(lines):
    """Lightweight FIR formatter: groups line-wise OCR into numbered sections."""
    sections = []
    for idx, text in enumerate(lines, start=1):
        sections.append({"section": f"Line {idx}", "text": text})
    return {"status": "draft", "lines": sections}

def predict(input_img, progress=gr.Progress()):
    """Full pipeline with visual stages and simple FIR formatting."""
    import time
    
    if input_img is None:
        return "", None, None, [], [[]], {}, "❌ No image provided"
    
    status_msg = ""
    print(f"Device: {device} | GPU Available: {torch.cuda.is_available()}")
    
    # Stage 1: Detection (ultra-aggressive)
    progress(0, desc="🔍 Detecting text lines...")
    status_msg += "🔍 Stage 1: Detecting text lines...\n"
    start = time.time()
    detection_results = detection_model.predict(
        source=input_img, 
        conf=0.01,  # Ultra-low confidence to catch everything
        imgsz=1280,  # Maximum size for best detection
        save=False, 
        agnostic_nms=True,  # Class-agnostic NMS
        max_det=300,  # Allow up to 300 detections
        device=device,
        verbose=False
    )
    elapsed = time.time()-start
    
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    bounding_boxes.sort(key=lambda x: x[1])
    status_msg += f"   ✓ Found {len(bounding_boxes)} lines in {elapsed:.2f}s\n"
    print(status_msg)
    
    progress(0.2, desc=f"🎨 Drawing boxes on {len(bounding_boxes)} lines...")
    
    # Stage 2: Overlay visualization
    overlay = input_img.copy()
    draw = ImageDraw.Draw(overlay)
    from numpy import random
    for i, box in enumerate(bounding_boxes):
        color = tuple(random.randint(50, 255, 3))
        draw.rectangle(box, outline=color, width=4)
        draw.text((box[0] + 5, box[1] + 5), str(i + 1), fill=color)
    status_msg += "   ✓ Overlay created\n"
    
    progress(0.3, desc="✂️ Cropping lines...")
    
    # Stage 3: Crops
    cropped_images = [input_img.crop(box) for box in bounding_boxes]
    status_msg += f"   ✓ Cropped {len(cropped_images)} lines\n"
    
    # Prep early gallery for visual feedback
    gallery_items = [(img, f"Line {idx}") for idx, img in enumerate(cropped_images, 1)]
    
    progress(0.4, desc="📖 Running OCR...")
    
    # Stage 4: OCR per crop
    line_texts = []
    start = time.time()
    for idx, img in enumerate(cropped_images, 1):
        progress(0.4 + (0.5 * idx / len(cropped_images)), desc=f"📖 OCR Line {idx}/{len(cropped_images)}...")
        try:
            text = text_recognizer(img, recognition_model, converter, device)
            line_texts.append(text)
            print(f"   Line {idx}: {text[:60]}")
        except Exception as e:
            print(f"   Line {idx}: ERROR - {e}")
            line_texts.append("[ERROR]")
    
<<<<<<< Updated upstream
    elapsed = time.time()-start
    status_msg += f"   ✓ OCR completed in {elapsed:.2f}s\n"

    joined_text = "\n".join(line_texts)
    
    progress(0.95, desc="📋 Formatting FIR...")
=======
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
>>>>>>> Stashed changes
    
    # Stage 5: Structured FIR
    fir_struct = format_fir(line_texts)
    status_msg += "   ✓ FIR formatted\n"

    # Update gallery with OCR text
    gallery_items_final = []
    for idx, img in enumerate(cropped_images, start=1):
        txt = line_texts[idx-1] if idx-1 < len(line_texts) else ''
        gallery_items_final.append((img, f"L{idx}: {txt[:80]}"))
    
    status_msg += "✅ DONE! All stages completed.\n"
    print(status_msg)
    
    progress(1.0, desc="✅ Complete!")
    
    return joined_text, overlay, gallery_items_final, gallery_items_final, [[i+1, t] for i, t in enumerate(line_texts)], fir_struct, status_msg

<<<<<<< Updated upstream
with gr.Blocks(title="🌙 Urdu OCR - UTRNet Dashboard", theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🌙 Urdu OCR - UTRNet Visual Pipeline\nUpload FIR or any Urdu document to extract text with live processing feedback.")
    
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(
                type="pil",
                label="📄 Upload Document",
                sources=["upload", "clipboard"],
                height=500
=======
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
>>>>>>> Stashed changes
            )
            run_btn = gr.Button("▶️ Run OCR", size="lg", variant="primary")
            
            gr.Markdown("### 📋 Example Images")
            gr.Examples(
                examples,
                inputs=inp,
                label="Click to try examples"
            )
        
        with gr.Column(scale=2):
            status_box = gr.Textbox(label="⚙️ Processing Status", lines=10, max_lines=15, interactive=False)
            recognized = gr.Textbox(label="📝 Recognized Text (Urdu)", lines=12)
            struct = gr.JSON(label="📋 Formatted FIR Output")
    
    with gr.Accordion("🔧 Processing Details (Click to Expand)", open=False):
        det_img = gr.Image(label="🎯 Detection Overlay with Numbered Boxes", type="pil")
        
        gallery = gr.Gallery(
            label="✂️ Cropped Lines with OCR Preview",
            columns=5,
            height="auto",
            show_label=True
        )
        
        line_table = gr.Dataframe(
            headers=["Line #", "Urdu Text"],
            label="📖 Line-by-Line OCR Results",
            wrap=True,
            interactive=False
        )

<<<<<<< Updated upstream
    run_btn.click(
        fn=predict,
        inputs=inp,
        outputs=[recognized, det_img, gallery, gallery, line_table, struct, status_box]
    )
=======
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
>>>>>>> Stashed changes

if __name__ == "__main__":
    iface.launch()