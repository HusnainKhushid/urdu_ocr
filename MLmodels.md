Top 3 Recommended ML Models for Urdu OCR
These three models are selected for being easiest to implement, trivial to train (or unsupervised), and having low risk (modular "plug-and-play" components).

1. Linear Regression (Preprocessing)
Application: Document Skew Correction
Why it's easy: You don't need a dataset. You just use the coordinates of the pixels in the image itself.
Implementation:
Use sklearn.linear_model.LinearRegression.
Input: $(x, y)$ coordinates of all black pixels (text).
Output: Slope $m$ of the text lines.
Action: Rotate image by $-\tan^{-1}(m)$.
Risk: Near zero. If the document is already straight, the slope will be ~0, and it does nothing.
Location: 
src/utils.py
 (add correct_skew function).
2. K-Means Clustering (Segmentation/Preprocessing)
Application: adaptive Binarization (Background Removal)
Why it's easy: It is unsupervised. You don't need labeled "foreground/background" data. It learns from the image itself instantly.
Implementation:
Use sklearn.cluster.KMeans with n_clusters=2.
Input: Flattened image pixel values.
Output: Labels mapping every pixel to "Dark" (Text) or "Light" (Background).
Action: Replace all "Background" pixels with pure white (255) and "Text" with pure black (0) or keep original intensity.
Risk: Better than a fixed threshold (e.g., < 127) because it adapts to dark/light images automatically.
Location: 
src/read.py
 (replace manual thresholding or standard preprocessing).
3. Naive Bayes (Post-Processing)
Application: Basic Urdu Spell Checker / Language Model
Why it's easy: Training only requires a simple list of valid Urdu words (dictionary). It counts frequencies.
Implementation:
Use sklearn.naive_bayes.MultinomialNB (or write a simple probabilistic counter).
Train: Feed it a text file of Urdu literature.
Inference: If the OCR outputs a word not in the dictionary (e.g., "بـتـا_"), use the model to find the most probable valid word (e.g., "کتاب") that is close in edit distance.
Risk: Low. You can set it to only trigger when the OCR confidence is low. It doesn't touch the complex image processing pipeline.
Location: 
src/read.py
 (after getting preds_str).