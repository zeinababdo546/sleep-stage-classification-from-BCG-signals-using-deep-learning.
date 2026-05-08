
# # Sleep-Stage-Classification-using-ECG-based-IHR-Signals
##  Abstract
"Automated sleep stage classification is vital for accessible sleep health monitoring. This project develops a high-performance pipeline for classifying five sleep stages (Wake, N1, N2, N3, and REM) using only single-lead ECG signals. Our methodology integrates digital signal processing with a hybrid CNN-LSTM architecture, combining 1D-ResNet for spatial feature extraction and Bidirectional LSTM for temporal context modeling.

To enhance classification accuracy, we utilize a dual-domain approach, merging time-domain Instantaneous Heart Rate (IHR) with spectral frequency features (LF, HF,lf_nu, hf_nu, LF/HF ratio). Key challenges like class imbalance and signal noise are addressed through SMOTE oversampling and strategic data augmentation. Evaluated on the MESA dataset, the model achieves a competitive 63% accuracy, providing a computationally efficient and interpretable alternative to traditional multi-modal sleep staging systems."

## Pipeline

### 1. Signal Preprocessing & Cleaning
To ensure the highest accuracy in R-peak detection, we implemented:
* **Normalization:** Patient-wise Z-score scaling.
* **Band-pass Filtering:** (0.5 Hz – 40 Hz) to remove noise and baseline wander.
* **IHR Extraction:** Calculating Instantaneous Heart Rate from R-R intervals.
* **Resampling:** 4Hz Linear Interpolation for uniform data dimensions.



### 2. Feature Engineering & Data Balancing
* **Hybrid Features:** Integration of time-domain IHR with **Spectral Features** (LF, HF,lf_nu, hf_nu, LF/HF ratio) using Welch’s method.
* **Class Imbalance (SMOTE):** Addressed the scarcity of N1 and N3 stages using the Synthetic Minority Over-sampling Technique.
* **Augmentation:** Applied Gaussian noise and time-shifting to improve model generalization.

### 3. Model Architecture (CNN-LSTM)
We utilized a hybrid deep learning model to capture both spatial and temporal features:
* **Feature Extractor:** A 1D-CNN (ResNet) to identify morphological patterns in heart rate.
* **Temporal Context:** A **Bidirectional LSTM** that processes a sequence of **5 consecutive 30s windows**.



## Performance Results
The model achieved an overall accuracy of **63%**, a competitive result for ECG-only classification.

### Classification Report:
<img width="452" height="221" alt="image" src="https://github.com/user-attachments/assets/533af57f-5cab-4cae-a529-c45e39433d57" />

## Confusion Matrix
<img width="789" height="699" alt="image" src="https://github.com/user-attachments/assets/4e5d742e-81f1-4b09-96e7-a105b6b4a0e0" />

## Test accuracy




