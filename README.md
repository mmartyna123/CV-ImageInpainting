# 🫛 Image Inpainting of Beans 🫛




## 📌 Introduction
This project focuses on **image inpainting**, the process of restoring missing or damaged parts of an image by generating visually coherent content. The goal is to reconstruct a complete image given an input with missing regions.

We tested multiple architectures and techniques to achieve high-quality inpainting results with smooth blending and structural consistency.

---

##  Dataset
We used the **Vegetable Image Dataset** from Kaggle, specifically working with **"Beans"**, consisting of **1,400 images (224x224)**. 

Additionally, we collected **567 images** of beans from the internet for **evaluation purposes**, containing more noise and variability.


**Holes** were created by placing black rectangles in the centre, with corresponding binary masks used as model inputs.

---

## 🔬 Models & Architectures

### 1️⃣ **Autoencoder**
- Encoder **reduces spatial dimensions**, bottleneck compresses features.
- Decoder **upsamples and reconstructs** the missing region.
- Uses **Sigmoid activation** to keep pixel values in <0,1>.


---

### 2️⃣ **Autoencoder with Skip Connections**
- Improves reconstruction by **retaining spatial details** via skip connections.
- Skip connections **transfer encoder features** directly to the decoder.


✅ **Best model for inpainting** due to its balance of speed and performance.

---

### 3️⃣ **GAN (Generative Adversarial Network)**
- **Generator**: Encoder-decoder structure with skip connections.
- **Discriminator**: Convolutional network distinguishing real vs. generated images.
- **Losses used**:
  - **Masked MSE Loss** (for pixel accuracy)
  - **Adversarial Loss** (for realism)
  - **Total Variation Loss** (for smoothness)



---

## 🎯 Evaluation & Metrics
To evaluate our models, we used:
- **Masked MSE Loss** – Measures pixel-wise error only in the missing region.  
- **Peak Signal-to-Noise Ratio (PSNR)** – Higher values indicate better reconstruction.  
- **Adversarial Loss (for GANs)** – Encourages realistic textures.  

Cross-validation (**5-fold**) was performed to confirm model generalization.

---



## 📦 Requirements
All dependencies are listed in `requirements.txt`.  
🛠 **Key Libraries**:
- PyTorch, Torchvision
- MLflow, DVC, DVC Live
- TensorBoard
- NumPy, OpenCV, Matplotlib



## References
- [Kaggle: Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)  
- [Generative Adversarial Networks (AWS)](https://aws.amazon.com/what-is/gan/)  
- [Image Inpainting with Deep Learning](https://wandb.ai/wandb_fc/articles/reports/Introduction-to-image-inpainting-with-deep-learning--Vmlldzo1NDI3MjA5)  
- [OpenCV Documentation](https://docs.opencv.org/)  

---


