📅 Week 1: The Science Behind Images
Focus: Establishing a strong foundation in computer vision history and medical documentation.

Tasks Completed:
Lecture Analysis: Studied "The Ancient Secrets of Computer Vision" by Joseph Redmon (Creator of YOLO). Documented key insights on:

Human Vision vs. Computer Vision.

Color Spaces and Image Transforms.

The evolution of object detection frameworks.

Medical Research: Conducted research on standard healthcare reports (Radiology, Pathology, Operative) and the specific requirements of Medical Imaging Reports.

📅 Week 2: Neural Networks & Image Processing
Focus: Practical implementation of traditional image processing algorithms using Python.

Tasks Completed:
Image Processing Pipeline: Developed a notebook using OpenCV to perform:

Preprocessing: Grayscale conversion and histogram analysis.

Feature Extraction: Canny Edge Detection.

Restoration: Gaussian noise simulation and removal using Gaussian Blur filters.

Dataset Auditing: Explored the NIH Chest X-Ray and ISIC 2019 datasets to identify challenges like class imbalance, artifacts, and labeling consistency.

Tech Stack: OpenCV, NumPy, Matplotlib

📅 Week 3: CNNs, Darknet, and YOLO
Focus: Training modern Convolutional Neural Networks (CNNs) for real-world dermatological classification.

Tasks Completed:
Model Training: Trained a YOLOv8-Nano Classification model (yolov8n-cls) on the "Skin Computer Vision Model" dataset (7,200+ images).

Performance:

Accuracy: Achieved 97.7% Top-1 Accuracy in just 10 epochs.

Inference: Validated the model's ability to distinguish between Basal Cell Carcinoma, Melanoma, and Nevus.

Analysis: Generated confusion matrices to visualize class-wise performance and minimize false negatives in critical classes like Melanoma.

Tech Stack: Ultralytics YOLOv8, Roboflow, PyTorch, Google Colab (T4 GPU)

📅 Week 4: Case Study - TILs and TIGER GC
Focus: Applying AI to Histopathology and Oncology (Breast Cancer).

Tasks Completed:
Problem Statement: Analyzed the TIGER Grand Challenge, focusing on the automated scoring of Tumor-Infiltrating Lymphocytes (TILs) in breast cancer tissue.

Clinical Impact: Explored how AI can replace manual, subjective scoring by pathologists to better predict patient response to treatment.

Pipeline Design: Designed a training pipeline for the WSI ROI (Region of Interest) dataset to detect immune cells in complex histopathology slides.

Key Resources: TIGER Grand Challenge, TILs in Breast Cancer

🚀 How to Run
Environment: The .ipynb notebooks are designed for Google Colab.

Requirements:

pip install ultralytics roboflow opencv-python

Usage:

Clone the repository.

Navigate to the specific week's folder.

Open the notebook and run all cells (Ensure GPU is enabled for Week 3 & 4).

Submitted by: Kushagra Raghuvanshi
