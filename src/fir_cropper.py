
import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    import layoutparser as lp
except ImportError:
    print("Error: layoutparser is not installed.")
    print("Please install it: pip install layoutparser 'layoutparser[detectron2]'")
    exit(1)

def run_fir_cropper(input_dir, output_dir, threshold, device):
    """
    Main function to crop FIR sections using LayoutParser.
    """
    # 1. Initialize Model
    # Using the robust PubLayNet model: mask_rcnn_X_101_32x8d_FPN_3x
    print("Initializing LayoutParser model...")
    
    # Manually handle weights download to avoid iopath issues
    weights_url = "https://www.dropbox.com/s/57zjbwv6gh3srry/model_final.pth?dl=1"
    models_dir = os.path.join(os.getcwd(), "models")
    weights_path = os.path.join(models_dir, "publaynet_model_final.pth")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    if not os.path.exists(weights_path):
        print(f"Downloading model weights to {weights_path}...")
        import urllib.request
        try:
            # Add headers to mimic browser to avoid some download issues
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(weights_url, weights_path)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download weights manually: {e}")
            return

    try:
        model = lp.Detectron2LayoutModel(
            config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
            model_path=weights_path, # Force use of local weights
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", threshold],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            device=device
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Ensure detectron2 is installed correctly.")
        return

    # 2. Process Images
    input_path = Path(input_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [f for f in os.listdir(input_path) if Path(f).suffix.lower() in image_extensions]

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} images. Processing to {output_dir}...")

    for img_name in tqdm(images):
        img_full_path = str(input_path / img_name)
        image = cv2.imread(img_full_path)
        
        if image is None:
            print(f"Could not read {img_full_path}")
            continue

        # Detect layout
        # Convert to RGB for LayoutParser (which expects RGB input for detection usually, 
        # though detectron2 internally handles it. Safe to pass BGR from cv2 if consistency maintained,
        # but standard practice is RGB.)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        layout = model.detect(image_rgb)

        # Process each detected block
        for i, block in enumerate(layout):
            # Block contains coordinates and type
            # block.type is mapped from label_map (e.g. "Table")
            category = block.type
            score = block.score
            
            # Create output subfolder
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Crop image (block.crop_image expects the image array)
            # define pad if needed, here we take tight crop
            segment = block.crop_image(image) # Use original BGR image for saving
            
            # Save
            filename = f"{os.path.splitext(img_name)[0]}_{i}_score{score:.2f}.jpg"
            save_path = os.path.join(category_dir, filename)
            cv2.imwrite(save_path, segment)

    print("Processing complete.")

def main():
    parser = argparse.ArgumentParser(description="Crop sections from FIR images using LayoutParser")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing FIR images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for cropped sections")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist.")
        exit(1)
        
    run_fir_cropper(args.input, args.output, args.threshold, args.device)

if __name__ == "__main__":
    main()
