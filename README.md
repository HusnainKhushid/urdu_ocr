# 🌟 Urdu OCR - High-Resolution Urdu Text Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)

An advanced deep learning system for recognizing Urdu text in printed documents, featuring state-of-the-art preprocessing, recognition, and post-processing capabilities.

## 📜 About

This project implements **UTRNet (Urdu Text Recognition Network)** presented at ICDAR 2023, with enhanced machine learning preprocessing and spell correction capabilities. The system is specifically optimized for recognizing Urdu text from FIR (First Information Reports) and other printed documents.

**Paper**: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents"  
**Authors**: Abdur Rahman, Arjun Ghosh, Chetan Arora  
**Conference**: ICDAR 2023  
**License**: Creative Commons Attribution-NonCommercial 4.0 International

## ✨ Key Features

### 🔬 **3-Stage ML Pipeline**

1. **Preprocessing** - Adaptive image enhancement using unsupervised ML
   - Linear Regression for automatic skew correction
   - K-Means clustering (k=2) for adaptive binarization

2. **Recognition** - Deep neural network for text extraction
   - HRNet feature extraction for high-resolution processing
   - Double Bidirectional LSTM for sequence modeling
   - CTC/Attention-based prediction

3. **Post-Processing** - Intelligent spell correction
   - Naive Bayes probabilistic correction
   - KNN-based similarity matching
   - Hybrid approach combining both methods

### 🚀 **Additional Capabilities**

- **YOLOv8 Integration**: Automatic line detection for multi-line documents
- **Web Interface**: Interactive Gradio dashboard for easy testing
- **Multiple Architectures**: Support for VGG, ResNet, DenseNet, UNet variants
- **Flexible Training**: Customizable pipeline with various model configurations
- **RTL Support**: Proper handling of right-to-left Urdu text

## 📁 Project Structure

```
urdu_ocr/
├── src/                          # Core source code
│   ├── read.py                   # Inference with full pipeline
│   ├── train.py                  # Training script
│   ├── test.py                   # Validation & testing
│   ├── spell_checkers.py         # 3 spell correction algorithms
│   ├── utils.py                  # Utilities (converters, skew correction)
│   └── dataset.py                # Data loading & augmentation
├── models/                       # Model architecture
│   ├── model.py                  # Main UTRNet model
│   └── modules/                  # Neural network modules
├── Dashboard/                    # Web interface
│   └── dashboard_app/
│       └── app.py                # Gradio application
├── notebooks/                    # Jupyter notebooks for experimentation
├── FIR/                          # Sample FIR images
├── requirements.txt              # Python dependencies
└── MLmodels.md                   # ML implementation guide

```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/HusnainKhushid/urdu_ocr.git
cd urdu_ocr
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download required data** (if not included)
   - Pre-trained model weights: `models/best_norm_ED.pth`
   - YOLOv8 model: `models/yolov8m_UrduDoc.pt`
   - Urdu glyphs file: `data/UrduGlyphs.txt`
   - Urdu literature corpus for spell checking (optional)

## 🚀 Quick Start

### Using the Web Dashboard

```bash
cd Dashboard
python dashboard_app/app.py
```

Then open your browser at `http://localhost:7860`

### Command-Line Inference

```python
from src.read import read
import argparse

# Basic usage
python src/read.py \
  --image_path "path/to/urdu_image.jpg" \
  --saved_model "models/best_norm_ED.pth" \
  --character "$(cat data/UrduGlyphs.txt)" \
  --correction_algo "hybrid"
```

### Training a Model

```bash
python src/train.py \
  --train_data "data/train_lmdb" \
  --valid_data "data/valid_lmdb" \
  --select_data "UrduDoc" \
  --batch_size 32 \
  --FeatureExtraction "HRNet" \
  --SequenceModeling "DBiLSTM" \
  --Prediction "CTC"
```

## 📊 Model Architecture

### Feature Extraction Options
- **HRNet** (High-Resolution Network) - Default, best performance
- VGG, ResNet, DenseNet
- UNet, UNet++, ResUNet, AttnUNet, InceptionUNet
- RCNN

### Sequence Modeling Options
- **DBiLSTM** (Double Bidirectional LSTM) - Default
- BiLSTM, LSTM, GRU, MDLSTM

### Prediction Options
- **CTC** (Connectionist Temporal Classification) - Default
- Attention mechanism

## 🧪 Testing & Evaluation

### Test on a dataset
```bash
python src/test.py \
  --eval_data "data/test_lmdb" \
  --saved_model "models/best_norm_ED.pth"
```

### Character-level testing
```bash
python src/char_test.py \
  --eval_data "data/test_lmdb" \
  --saved_model "models/best_norm_ED.pth"
```

## 📈 Performance Enhancement

### Spell Correction Algorithms

1. **Naive Bayes** - Fast, probabilistic correction based on Urdu literature
2. **KNN** - Character n-gram similarity with Levenshtein distance
3. **Hybrid** - Combines both for maximum accuracy

Enable spell correction:
```python
python src/read.py \
  --image_path "image.jpg" \
  --correction_algo "hybrid" \
  --literature_path "data/urdu_corpus.txt"
```

### Preprocessing Features

- **Automatic Skew Correction**: Detects and corrects document rotation
- **Adaptive Binarization**: Works with various lighting conditions
- **Smart Resizing**: Maintains aspect ratio while meeting model requirements

## 📝 Dataset Format

Create LMDB dataset from your images:

```bash
python src/create_lmdb_dataset.py \
  --inputPath "images/" \
  --gtFile "labels.txt" \
  --outputPath "data/train_lmdb"
```

Label format (labels.txt):
```
image1.jpg	اردو متن یہاں
image2.jpg	یہ ایک مثال ہے
```

## 🎯 Use Cases

- **Document Digitization**: Convert printed Urdu documents to editable text
- **FIR Processing**: Extract information from police First Information Reports
- **Archive Digitization**: Digitize Urdu newspapers, books, and historical documents
- **Form Processing**: Extract data from Urdu forms and applications
- **Educational Tools**: Assist in Urdu language learning applications

## 🔬 Technical Details

### Preprocessing
- **Skew Correction**: Uses linear regression on text pixel coordinates
- **Binarization**: K-Means clustering (unsupervised) for adaptive thresholding

### Training Details
- **Optimizer**: Adam with custom learning rate scheduling
- **Data Augmentation**: Random rotation, scaling, and distortion
- **Batch Balancing**: Hierarchical dataset sampling
- **Multi-GPU**: Support for distributed training

### Inference Optimizations
- Model caching for spell checkers
- Efficient LMDB reading
- GPU acceleration with CUDA

## 📦 Dependencies

Core libraries:
- PyTorch >= 2.0
- torchvision
- OpenCV (opencv-python, opencv-contrib-python)
- Ultralytics (YOLOv8)
- Gradio (Web UI)
- scikit-learn (ML algorithms)
- PyArabic, arabic-reshaper (Urdu text handling)

See [requirements.txt](requirements.txt) for complete list.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.  
See [LICENSE](http://creativecommons.org/licenses/by-nc/4.0/) for details.

**Note**: This is for non-commercial use only. For commercial licensing, please contact the original authors.

## 🙏 Acknowledgments

- Original UTRNet paper authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
- ICDAR 2023 conference
- PyTorch and Ultralytics communities
- Contributors and testers

## 📧 Contact

For questions, issues, or collaboration:
- GitHub Issues: [Create an issue](https://github.com/HusnainKhushid/urdu_ocr/issues)
- Original Project: [UTRNet GitHub](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)

## 🔗 References

- [UTRNet Project Website](https://abdur75648.github.io/UTRNet/)
- [ICDAR 2023 Paper](https://abdur75648.github.io/UTRNet/)
- [HRNet Paper](https://arxiv.org/abs/1908.07919)

---

**Made with ❤️ for the Urdu NLP community**
