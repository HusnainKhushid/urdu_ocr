import torch
import gradio as gr
from .read import text_recognizer
from .model import Model
from .utils import CTCLabelConverter
from ultralytics import YOLO
from PIL import ImageDraw
import os


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

examples = [os.path.join(BASE_DIR, "static", "images", "1.jpg"),
            os.path.join(BASE_DIR, "static", "images", "2.jpg"),
            os.path.join(BASE_DIR, "static", "images", "3.jpg")]

input = gr.Image(type="pil",image_mode="RGB", label="Input Image")

def predict(input):
    "Line Detection"
    detection_results = detection_model.predict(source=input, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    bounding_boxes.sort(key=lambda x: x[1])
    
    "Draw the bounding boxes"
    draw = ImageDraw.Draw(input)
    for box in bounding_boxes:
        # draw rectangle outline with random color and width=5
        from numpy import random
        draw.rectangle(box, fill=None, outline=tuple(random.randint(0,255,3)), width=5)
    
    "Crop the detected lines"
    cropped_images = []
    for box in bounding_boxes:
        cropped_images.append(input.crop(box))
    len(cropped_images)
    
    "Recognize the text"
    texts = []
    for img in cropped_images:
        texts.append(text_recognizer(img, recognition_model, converter, device))
    
    return "\n".join(texts), input

output_image = gr.Image(type="pil",image_mode="RGB",label="Detected Lines")
output_text = gr.Textbox(label="Recognized Text",interactive=True)

iface = gr.Interface(fn=predict, 
                     inputs=input, 
                     outputs=[gr.Textbox(label="Recognized Text"), gr.Image(label="Detection Result")],
                     title="Urdu Text Recognition",
                     description="An application to recognize text from images of Urdu text.",
                     examples=examples)

if __name__ == "__main__":
    iface.launch()