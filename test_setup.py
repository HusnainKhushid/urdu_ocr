print("Starting test...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    
    import gradio
    print(f"✓ Gradio {gradio.__version__}")
    
    from ultralytics import YOLO
    print("✓ Ultralytics YOLO")
    
    import os
    print(f"✓ Working directory: {os.getcwd()}")
    
    # Test file paths
    vocab_file = "data/UrduGlyphs.txt"
    model_file = "models/best_norm_ED.pth"
    yolo_file = "models/yolov8m_UrduDoc.pt"
    
    print(f"\nFile checks:")
    print(f"  Vocab: {os.path.exists(vocab_file)} - {vocab_file}")
    print(f"  Model: {os.path.exists(model_file)} - {model_file}")
    print(f"  YOLO:  {os.path.exists(yolo_file)} - {yolo_file}")
    
    print("\n✓ All basic checks passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
