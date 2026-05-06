# sleep-stage-classification-from-EKG-signals-using-deep-learning.
# Sleep Stage Classification using Temporal ECG Analysis
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![MNE](https://img.shields.io/badge/MNE--Python-0.24+-green.svg)

# Temporal Sleep Stage Classification using ECG-based CNN-LSTM

##  Abstract
This project focuses on classifying sleep stages (**Wake, N1, N2, N3, and REM**) using only **single-lead ECG signals**. By analyzing Heart Rate Variability (HRV) and temporal patterns, we avoid the complexity of traditional EEG-based Polysomnography.

## Pipeline

### 1. Signal Preprocessing & Cleaning
To ensure the highest accuracy in R-peak detection, we implemented:
* **Band-pass Filtering:** (0.5 Hz – 40 Hz) to remove noise and baseline wander.
* **IHR Extraction:** Calculating Instantaneous Heart Rate from R-R intervals.
* **Resampling:** 4Hz Linear Interpolation for uniform data dimensions.
* **Normalization:** Patient-wise Z-score scaling.



### 2. Feature Engineering & Data Balancing
* **Hybrid Features:** Integration of time-domain IHR with **Spectral Features** (LF, HF, LF/HF ratio) using Welch’s method.
* **Class Imbalance (SMOTE):** Addressed the scarcity of N1 and N3 stages using the Synthetic Minority Over-sampling Technique.
* **Augmentation:** Applied Gaussian noise and time-shifting to improve model generalization.

### 3. Model Architecture (CNN-LSTM)
We utilized a hybrid deep learning model to capture both spatial and temporal features:
* **Feature Extractor:** A 1D-CNN (ResNet) to identify morphological patterns in heart rate.
* **Temporal Context:** A **Bidirectional LSTM** that processes a sequence of **5 consecutive 30s windows**.



## Performance Results
The model achieved an overall accuracy of **63%**, a competitive result for ECG-only classification.

### Classification Report:
| Stage | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **Wake** | 0.78      | 0.83   | 0.81     |
| **N1** | 0.19      | 0.13   | 0.15     |
| **N2** | 0.58      | 0.70   | 0.63     |
| **N3** | 0.37      | 0.21   | 0.27     |
| **REM** | 0.42      | 0.22   | 0.29     |



## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure EDF and XML files are in the designated `/Dataset` folder.
3. Run the notebook: `Sleep_Classification.ipynb`.
