WIDS

## 📅 Week 1: Neural Networks & Image Processing

**Focus:** Understanding the basics of how computers "see" images and manipulating pixel data using Python.

### Tasks Completed:
* **Image Manipulation:** Loaded medical X-rays and performed grayscale conversion, resizing, and histogram analysis.
* **Edge Detection:** Implemented Canny Edge Detection to identify structural boundaries in X-rays.
* **Noise Handling:** Added Gaussian noise to images and restored them using Gaussian Blurring filters.
* **Dataset Research:** Analyzed two major medical datasets (Chest X-Ray Pneumonia & ISIC Skin Lesion) to understand class balance and data challenges.

**Tools Used:** `OpenCV`, `NumPy`, `Matplotlib`

---

## 📅 Week 2: CNNs, Darknet, and YOLO

**Focus:** Training state-of-the-art Convolutional Neural Networks (CNNs) for real-world classification tasks.

### Tasks Completed:
* **Dataset Acquisition:** Integrated the Roboflow API to source the "Skin Computer Vision Model" dataset.
* **Model Training:** Trained a **YOLOv8-Nano Classification** model (`yolov8n-cls`) on 7,200+ images.
* **Performance:**
    * **Epochs:** 10
    * **Top-1 Accuracy:** **97.7%**
    * **Loss:** Converged successfully.
* **Evaluation:** Generated confusion matrices to visualize prediction performance across classes (Basal Cell Carcinoma, Melanoma, Nevus).

**Tools Used:** `Ultralytics YOLOv8`, `Roboflow`, `PyTorch`

---

## 🚀 How to Run

1.  **Open Notebooks:** The `.ipynb` files are designed to run in **Google Colab**.
2.  **Dependencies:**
    * Week 1 requires `opencv-python`.
    * Week 2 requires `ultralytics` and `roboflow`.
3.  **Hardware:** Week 2 training is optimized for a T4 GPU runtime.

---

*Submitted by: Kushagra Raghuvanshi*

