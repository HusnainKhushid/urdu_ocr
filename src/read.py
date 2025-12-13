"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import pytz
import math
import argparse
from PIL import Image
from datetime import datetime

import torch
import torch.utils.data

from models.model import Model
from src.dataset import NormalizePAD
from src.dataset import NormalizePAD
from src.dataset import NormalizePAD
from src.utils import CTCLabelConverter, AttnLabelConverter, Logger, correct_skew
from src.spell_checkers import NaiveBayesCorrector, KNNCorrector, HybridCorrector
from sklearn.cluster import KMeans
from collections import Counter
import re
import numpy as np
import pickle

def adaptive_binarization(image):
    """
    Applies adaptive binarization using K-Means Clustering (K=2) on pixel intensities.
    Input: PIL Image
    Output: Binarized PIL Image
    """
    # Convert to grayscale numpy array
    try:
        if image.mode != 'L':
            image = image.convert('L')
    except Exception:
        pass # Handle if it's already a tensor or something unexpected, but we expect PIL

    img_array = np.array(image)
    h, w = img_array.shape
    pixels = img_array.reshape(-1, 1) # Flatten
    
    # Use MiniBatchKMeans for speed if available, else KMeans
    # User asked for sklearn.cluster.KMeans
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Identify which cluster is background (lighter, higher value)
    if centers[0] > centers[1]:
        bg_label = 0
        text_label = 1
    else:
        bg_label = 1
        text_label = 0
        
    # Replace background with 255 (white) and text with 0 (black)
    # Or keep original intensity for text? User said: "Replace all 'Background' pixels with pure white (255) and 'Text' with pure black (0) or keep original intensity."
    # Binarization usually implies 0/255. Let's do 0/255 for clean OCR.
    
    binarized_pixels = np.where(labels == bg_label, 255, 0).astype(np.uint8)
    binarized_img_array = binarized_pixels.reshape(h, w)
    
    return Image.fromarray(binarized_img_array)


def read(opt, device):
    opt.device = device
    os.makedirs("read_outputs", exist_ok=True)
    datetime_now = str(datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H-%M-%S"))
    logger = Logger(f'read_outputs/{datetime_now}.txt')
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    logger.log('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = model.to(device)

    # load model
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    logger.log('Loaded pretrained model from %s' % opt.saved_model)
    model.eval()
    
    if opt.rgb:
        img = Image.open(opt.image_path).convert('RGB')
    else:
        img = Image.open(opt.image_path).convert('L')
    
    # 1. Linear Regression Skew Correction
    try:
        img = correct_skew(img)
    except Exception as e:
        print(f"Warning: Skew correction failed: {e}")

    # 2. K-Means Adaptive Binarization
    # Only if not RGB (grayscale)
    if not opt.rgb:
        try:
             # Need to ensure we pass a PIL image or handle conversion
            img = adaptive_binarization(img)
        except Exception as e:
            print(f"Warning: Binarization failed: {e}")

    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(opt.imgH * ratio) > opt.imgW:
        resized_w = opt.imgW
    else:
        resized_w = math.ceil(opt.imgH * ratio)
    img = img.resize((resized_w, opt.imgH), Image.Resampling.BICUBIC)
    transform = NormalizePAD((1, opt.imgH, opt.imgW))
    img = transform(img)
    img = img.unsqueeze(0)
    # print(img.shape) # torch.Size([1, 1, 32, 400])
    batch_size = img.shape[0] # 1
    img = img.to(device)
    preds = model(img)
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
    
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    
    # 3. Spell Checker (Post-Processing)
    if opt.correction_algo != 'none' and opt.literature_path:
        checker = None
        if opt.correction_algo == 'naive_bayes':
            checker = NaiveBayesCorrector(opt.literature_path, alphabet=opt.character)
        elif opt.correction_algo == 'knn':
            checker = KNNCorrector(opt.literature_path)
        elif opt.correction_algo == 'hybrid':
            checker = HybridCorrector(opt.literature_path)
            
        if checker:
            preds_str = checker.correct_text(preds_str)

    logger.log(preds_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, help='path to image to read')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=100, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=400, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    """ Model Architecture """
    parser.add_argument('--FeatureExtraction', type=str, default="HRNet", #required=True,
                        help='FeatureExtraction stage VGG|RCNN|ResNet|UNet|HRNet|Densenet|InceptionUnet|ResUnet|AttnUNet|UNet|VGG')
    parser.add_argument('--SequenceModeling', type=str, default="DBiLSTM", #required=True,
                        help='SequenceModeling stage LSTM|GRU|MDLSTM|BiLSTM|DBiLSTM')
    parser.add_argument('--Prediction', type=str, default="CTC", #required=True,
                        help='Prediction stage CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    """ GPU Selection """
    parser.add_argument('--device_id', type=str, default=None, help='cuda device ID')
    parser.add_argument('--literature_path', type=str, default='data/urdu_words.txt', help='Path to Urdu literature text file')
    parser.add_argument('--correction_algo', type=str, default='naive_bayes', choices=['none', 'naive_bayes', 'knn', 'hybrid'], help='Spell correction algorithm')
    # parser.add_argument('--use_knn', action='store_true', help='Deprecated: use --correction_algo knn')
    
    opt = parser.parse_args()
    if opt.FeatureExtraction == "HRNet":
        opt.output_channel = 32
    """ vocab / character number configuration """
    file = open("data/UrduGlyphs.txt","r",encoding="utf-8")
    content = file.readlines()
    content = ''.join([str(elem).strip('\n') for elem in content])
    opt.character = content+" "
    
    cuda_str = 'cuda'
    if opt.device_id is not None:
        cuda_str = f'cuda:{opt.device_id}'
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)
    
    read(opt, device)