import os
import json
import torch
import gradio as gr
from ultralytics import YOLO
from PIL import ImageDraw

from .read import text_recognizer
from .model import Model
from .utils import CTCLabelConverter


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
    
    elapsed = time.time()-start
    status_msg += f"   ✓ OCR completed in {elapsed:.2f}s\n"

    joined_text = "\n".join(line_texts)
    
    progress(0.95, desc="📋 Formatting FIR...")
    
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

with gr.Blocks(title="🌙 Urdu OCR - UTRNet Dashboard") as iface:
    gr.Markdown("# 🌙 Urdu OCR - UTRNet Visual Pipeline\nUpload FIR or any Urdu document to extract text with live processing feedback.")
    
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(
                type="pil",
                label="📄 Upload Document",
                sources=["upload", "clipboard"],
                height=500
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

    run_btn.click(
        fn=predict,
        inputs=inp,
        outputs=[recognized, det_img, gallery, gallery, line_table, struct, status_box]
    )

if __name__ == "__main__":
    iface.launch()