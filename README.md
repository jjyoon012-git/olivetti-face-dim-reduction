# 🎅 Olivetti Face Dimensionality Reduction

## 1. Introduction

This project compares **PCA**, **LDA**, and **Autoencoder** for dimensionality reduction on facial image data.  
Although all three compress high-dimensional data into low-dimensional representations, their optimization objectives and statistical interpretations differ.  
Additionally, an Autoencoder was implemented based on Chapter 8 of the reference material to explore nonlinear dimensionality reduction.

### 1.1 PCA and LDA

**Principal Component Analysis (PCA)** is an **unsupervised linear dimensionality reduction** technique that preserves the directions of maximum variance in the data.  
It finds a projection that minimizes reconstruction error, making it highly effective for **information preservation and data compression**, such as in image denoising and face representation.

**Linear Discriminant Analysis (LDA)**, on the other hand, is a **supervised method** that finds projection directions maximizing class separability — bringing samples of the same class closer and pushing different classes apart.  
LDA is optimized for classification tasks but limited by the number of classes (max dimension = #classes − 1).

### 1.2 Why Autoencoder? (Nonlinear Dimensionality Reduction)

Both PCA and LDA assume linearity, which limits their ability to model **nonlinear variations** such as lighting, facial angle, and subtle expressions.  
To address this, a **deep learning-based Autoencoder** was employed. It encodes an image into a low-dimensional latent space and reconstructs it through decoding, automatically learning nonlinear structures within the data.

Thus, for complex datasets like faces, Autoencoders can potentially achieve **better reconstruction quality** and richer representations than linear methods.

### 1.3 Olivetti Faces Dataset

Source: [Kaggle – Olivetti Faces Dataset](https://www.kaggle.com/datasets/sahilyagnik/olivetti-faces)

The dataset contains **400 grayscale face images** from **40 individuals (10 per person)**, each sized **64×64 (4096 features)**.  
It includes variations in lighting, facial orientation, and expression.  
Since the feature dimension (4096) far exceeds the number of samples (400), it represents a **High Dimension, Low Sample Size (HDLSS)** problem — ideal for testing dimensionality reduction techniques.

---

## 2. Method

The study compared **PCA**, **LDA**, and **Autoencoder** on the same dataset to analyze reconstruction quality and classification performance.

### 2.1 Preprocessing and Dataset Split

1. Load the dataset using `sklearn.datasets.fetch_olivetti_faces()`.
2. Split into **train/test** using **stratified sampling** to preserve per-person balance.
3. Standardize data using **mean-centering** and **StandardScaler** for consistent variance scaling.

### 2.2 Dimensionality Reduction Pipelines

- **PCA (Unsupervised, Linear):** Projects onto directions of maximal variance for minimal information loss.
- **LDA (Supervised, Linear):** Projects onto directions that maximize class separability.
- **Autoencoder (Unsupervised, Nonlinear):** Neural network encoding and decoding facial structures.

### 2.2.1 Autoencoder Architecture

- **Input:** 64×64 grayscale (flattened to 4096)  
- **Encoder:** Lightweight CNN layers  
- **Latent Dimension:** 32  
- **Decoder:** Reconstructs input from latent features  
- **Loss:** MSE reconstruction loss  
- **Regularization:** Dropout, Early Stopping, Weight Decay  

All models shared identical preprocessing and data splits for fair comparison.

### 2.3 Classifier and Metrics

A **Linear SVM** (C = 1.0) was used to evaluate the quality of the low-dimensional embeddings.  
Metrics included:
- **Classification Accuracy**
- **Reconstruction Visualization**
- **Latent Space Clustering (t-SNE)**

---

## 3. Experiments

All experiments were conducted in **Jupyter Notebook (.ipynb)** format.

### 3.1 Experimental Settings

- Total Samples: 400  
- Split: 80% train / 20% test (stratified)  
- Standardization applied  
- Dimensions tested:
  - PCA = 32
  - LDA = 32 (out of 39 possible)
  - Autoencoder latent = 32  
- Classifier: Linear SVM (C = 1.0)

### 3.2 Visualization and Key Analyses

- **PCA:** Captured lighting and global facial structure  
- **LDA:** Highlighted class-specific discriminative features  
- **Autoencoder:** Latent space t-SNE revealed natural clustering

---

## 4. Results & Discussion

### **4.1 Eigenfaces & Fisherfaces**

#### PCA – Eigenfaces  
![Eigenfaces (PCA)](results/eigenfaces(pca).png)

#### LDA – Fisherfaces  
![Fisherfaces (LDA)](results/fisherfaces(lda).png)

- **PCA (Eigenfaces):** Represent dominant global patterns (lighting, contour).  
- **LDA (Fisherfaces):** Maximizes inter-class variance; however, due to HDLSS nature, results appear noisy.

---

### **4.2 Latent Space Analysis via t-SNE**

![t-SNE (Autoencoder)](results/tsne(ae).png)

Autoencoder’s latent vectors formed well-separated clusters per subject, despite being unsupervised — showing its ability to learn identity features implicitly.

---

### **4.3 Reconstruction Quality Comparison**

![Reconstruction (PCA vs AE)](results/reconstruction(pca,ae).png)

| Metric | PCA (32D) | Autoencoder (32D) |
|--------|------------|------------------|
| **MSE** | 0.0041 | 0.0053 |

- **PCA:** Lower MSE; smooth global reconstruction but slightly blurred details.  
- **AE:** Slightly higher MSE, showing less consistency, though capable of capturing nonlinear texture details.

---

### **4.4 Reconstruction Error Distribution**

![Reconstruction Error Histogram](results/reconhistogram(pca,ae).png)

PCA’s errors were tightly clustered (low variance), while AE showed a broader error spread — suggesting sample-dependent reconstruction difficulty.

---

### **4.5 Classification Accuracy (Linear SVM)**

![SVM Accuracy Comparison](results/linearsvm(pca,lda,ae).png)

| Method | Accuracy |
|--------|-----------|
| LDA | **1.000** |
| PCA | 0.963 |
| Autoencoder | 0.938 |

LDA achieved perfect accuracy due to label-guided optimization, while PCA and AE lagged slightly behind.

---

### **4.6 Effect of Latent Dimension**

![Latent Dimension Comparison](results/latentdim(pca,ae).png)

| Method | 8D | 16D | 32D |
|--------|----|-----|-----|
| PCA + SVM | 0.875 | 0.925 | **0.963** |
| AE + SVM | 0.863 | 0.913 | 0.925 |

Higher dimensions yielded better accuracy; PCA consistently outperformed AE across all latent sizes.

---

### **4.7 Intra-class / Inter-class Distance Analysis**

![Intra vs Inter Distance](results/intrainter(pca,lda,ae).png)

| Method | Intra | Inter |
|--------|--------|--------|
| PCA | 45.76 | 80.73 |
| LDA | **7.68** | **24.24** |
| AE | 9.16 | 16.90 |

- **LDA:** Small intra-class and large inter-class distances → strongest separability  
- **AE:** Moderate separability (unsupervised learning)
- **PCA:** Preserves overall variance but weak identity clustering  

**Overall separability ranking:** LDA > AE > PCA

---

## 5. Conclusion

### 5.1 PCA
Stable reconstruction and effective compression.  
However, as an unsupervised method, it lacks discriminative power — leading to moderate classification accuracy.

### 5.2 LDA
Best performance in classification by leveraging label information.  
However, its reconstructions are less interpretable due to its class-separability focus.

### 5.3 Autoencoder
Balances reconstruction quality and discriminative representation.  
Although its classification accuracy is lower, it successfully captures nonlinear face features without supervision.

---
## Author

**Written by:** Jeong-Ah Yoon
**GitHub:** [jjyoon012-git](https://github.com/jjyoon012-git)
