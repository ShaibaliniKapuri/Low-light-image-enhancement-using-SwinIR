# Low-Light Image Enhancement using SwinIR

This repository contains a solution for the **"Low-Light Image Denoising and Super-Resolution"** Kaggle competition. The goal is to take low-resolution, noisy images and transform them into high-resolution, clean outputs.

This project leverages the **SwinIR** model, a state-of-the-art image restoration architecture based on Swin Transformers, to perform **4x super-resolution and denoising**.

-----

## 📜 Project Overview

Low-light images often suffer from significant noise and a lack of detail, making them difficult to use. This competition challenges participants to build a model that can tackle two problems simultaneously:

1.  **Denoising:** Removing random variations and artifacts from the images.
2.  **Super-Resolution (4x):** Increasing the image dimensions by a factor of four, adding fine details and improving clarity.

Submissions are evaluated based on the **Peak-Signal-to-Noise-Ratio (PSNR)**, which measures the quality of the enhanced image against a clean, high-resolution ground truth. A higher PSNR score indicates a better restoration.

-----

## 🤖 Model & Methodology

This solution uses a pre-trained **SwinIR** model for inference, avoiding the need for training from scratch. SwinIR has demonstrated excellent performance on various image restoration tasks, including real-world super-resolution.

### The Model

  - **SwinIR (Swin Transformer for Image Restoration)**: A powerful deep learning model that applies the efficiency and global context-awareness of Vision Transformers to image restoration.
  - **Pre-trained Weights**: We use the `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` model, which was pre-trained on diverse real-world images for 4x super-resolution. This makes it highly effective at handling complex noise patterns and generating realistic details.

### Inference Workflow

The process, detailed in the `swinir-nb1.ipynb` notebook, is as follows:

1.  **Environment Setup**: The official SwinIR GitHub repository is cloned to provide the necessary model architecture and utility functions.
2.  **Model Initialization**: The SwinIR model is defined and the pre-trained weights for 4x real-world super-resolution are downloaded and loaded.
3.  **Data Processing Loop**:
      * The script iterates through every low-resolution image in the competition's test set.
      * Each image is loaded and converted from a BGR `OpenCV` array to an RGB `PyTorch` tensor.
4.  **SwinIR Inference**: The prepared tensor is fed into the SwinIR model, which outputs the enhanced, high-resolution image.
5.  **Output Generation**: The resulting tensor is converted back into an image format (`uint8`) and saved to the results directory. These generated images are then used for submission to Kaggle.

-----

## ✨ Results

The effectiveness of the SwinIR model is evident in its ability to dramatically improve image quality. Below is a conceptual example of the transformation.

| Low-Resolution Noisy Input | High-Resolution Denoised Output |
| :---: | :---: |
|  |  |

The final performance is determined by the PSNR score calculated on the hidden test set by the Kaggle platform.

-----

## 🚀 How to Run

To reproduce the results and generate the output images:

1.  **Clone this Repository**:

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install Dependencies**: This project requires Python and several packages. The official SwinIR repository is also a key dependency.

    ```bash
    # Clone the original SwinIR repo for model code
    git clone https://github.com/JingyunLiang/SwinIR.git

    # Install Python packages
    pip install torch opencv-python numpy
    ```

3.  **Download Dataset**:

      * Download the competition dataset from the Kaggle data page.
      * Place the `testset` folder into the appropriate input directory structure (e.g., `/kaggle/input/dlp-iit-madras-2025/testset/`).

4.  **Execute the Notebook**:

      * Open and run the `swinir-nb1.ipynb` notebook in a Jupyter environment.
      * **Important**: Ensure the paths in the notebook for the SwinIR repository (`import sys; sys.path.append('/path/to/SwinIR')`), the test set (`folder_test`), and the save directory (`save_dir`) are correct for your system.
      * Upon completion, the enhanced images will be saved in the specified output directory, ready for submission.

-----

## Acknowledgments

This project is built upon the fantastic work by the authors of SwinIR.

  - **Original Paper**: [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257) by Jingyun Liang et al.
  - **Official Repository**: [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
