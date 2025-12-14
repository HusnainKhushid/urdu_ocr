import torch
import gradio as gr
from ultralytics import YOLO
from PIL import ImageDraw
import os
import sys

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dashboard'))

from Dashboard.dashboard_app.read import text_recognizer
from Dashboard.dashboard_app.model import Model
from Dashboard.dashboard_app.utils import CTCLabelConverter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

""" vocab / character number configuration """
file = open(os.path.join(BASE_DIR, "data", "UrduGlyphs.txt"),"r",encoding="utf-8")
content = file.readlines()
content = ''.join([str(elem).strip('\n') for elem in content])
content = content+" "

""" model configuration """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
converter = CTCLabelConverter(content)
recognition_model = Model(num_class=len(converter.character), device=device)
recognition_model = recognition_model.to(device)
recognition_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "best_norm_ED.pth"), map_location=device))
recognition_model.eval()

detection_model = YOLO(os.path.join(BASE_DIR, "models", "yolov8m_UrduDoc.pt"))

# Check for example images
examples_dir = os.path.join(BASE_DIR, "Dashboard", "static", "images")
examples = []
if os.path.exists(examples_dir):
    examples = [os.path.join(examples_dir, f) for f in ["1.jpg", "2.jpg", "3.jpg"] if os.path.exists(os.path.join(examples_dir, f))]

input_image = gr.Image(type="pil",image_mode="RGB", label="Input Image")

def predict(input_img):
    """Line Detection"""
    detection_results = detection_model.predict(source=input_img, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    bounding_boxes.sort(key=lambda x: x[1])
    
    """Draw the bounding boxes"""
    draw = ImageDraw.Draw(input_img)
    for box in bounding_boxes:
        from numpy import random
        draw.rectangle(box, fill=None, outline=tuple(random.randint(0,255,3)), width=5)
    
    """Crop the detected lines"""
    cropped_images = []
    for box in bounding_boxes:
        cropped_images.append(input_img.crop(box))
    
    """Recognize the text"""
    texts = []
    for img in cropped_images:
        texts.append(text_recognizer(img, recognition_model, converter, device))
    
    return "\n".join(texts), input_img

iface = gr.Interface(
    fn=predict, 
    inputs=input_image, 
    outputs=[gr.Textbox(label="Recognized Text"), gr.Image(label="Detection Result")],
    title="Urdu Text Recognition - UTRNet",
    description="An application to recognize text from images of Urdu text using YOLOv8 detection and UTRNet recognition.",
    examples=examples if examples else None
)

if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Model loaded from: {os.path.join(BASE_DIR, 'models', 'best_norm_ED.pth')}")
    print(f"YOLO loaded from: {os.path.join(BASE_DIR, 'models', 'yolov8m_UrduDoc.pt')}")
    iface.launch(share=False)
