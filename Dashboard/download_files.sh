#!/bin/bash

# Create directories if they don't exist
mkdir -p static/images
mkdir -p models

# URLs of the files to be downloaded
image_urls=(
  "https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/1.jpg"
  "https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/2.jpg"
  "https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/3.jpg"
)

model_urls=(
  "https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/best_norm_ED.pth"
  "https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/yolov8m_UrduDoc.pt"
)

# Download image files
for url in "${image_urls[@]}"; do
  echo "Downloading $url..."
  curl -L -o static/images/$(basename "$url") "$url"
done

# Download model files
for url in "${model_urls[@]}"; do
  echo "Downloading $url..."
  curl -L -o models/$(basename "$url") "$url"
done

echo "All files downloaded successfully."
