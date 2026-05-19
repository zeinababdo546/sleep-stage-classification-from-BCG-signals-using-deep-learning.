
## Sleep-Stage-Classification-using-ECG-based-IHR-Signals
##  Abstract
Automated sleep stage classification is vital for accessible sleep health monitoring. This project develops a high-performance pipeline for classifying five sleep stages (Wake, N1, N2, N3, and REM) using only single-lead ECG signals. Our methodology integrates digital signal processing with a hybrid CNN-LSTM architecture, combining 1D-ResNet for spatial feature extraction and Bidirectional LSTM for temporal context modeling.

To enhance classification accuracy, we utilize a dual-domain approach, merging time-domain Instantaneous Heart Rate (IHR) with spectral frequency features (LF, HF,lf_nu, hf_nu, LF/HF ratio). Key challenges like class imbalance and signal noise are addressed through SMOTE oversampling and strategic data augmentation. Evaluated on the MESA dataset, the model achieves a competitive 63% accuracy, providing a computationally efficient and interpretable alternative to traditional multi-modal sleep staging systems.


## Tech Stack & Libraries
•	Language: Python 3.9
•	Biomedical Signal Processing: MNE-Python, SciPy (Signal Module)
•	Deep Learning Framework: PyTorch / TensorFlow
•	Machine Learning & Data Prep: Scikit-learn, Imbalanced-learn (SMOTE)
•	Data Manipulation: Pandas, NumPy
•	Parsing: xml.etree.ElementTree


## Detailed Project Pipeline
### 1. Signal Preprocessing & Cleaning
 
•	Bandpass Filtering: Raw EKG signals are filtered using a FIR design filter (0.5 Hz - 40.0 Hz) to eliminate baseline wander, powerline interference, and muscle artifacts.

•	R-Peak & IBI Extraction: R-peaks are isolated using mne.preprocessing.find_ecg_events. The differences between consecutive peaks generate the raw Inter-Beat Interval (IBI) series.

•	Outlier Rejection: Physiologically anomalous intervals (IBIs below 0.33s or above 1.5s, corresponding to >180 bpm or <40 bpm) are masked out. A Median Filter (medfilt with kernel size 3) smoothens the resulting Instantaneous Heart Rate (IHR).

### 2. Uniform Resampling & Window Epoching

•	Linear Interpolation: Because heartbeats occur at irregular intervals, the cleaned IHR is resampled onto a constant time grid at 4Hz using scipy.interpolate.interp1d.

•	30-Second Windowing: Continuous 4Hz data streams are segmented into 120-sample windows (30 seconds each) to perfectly align with standard clinical sleep staging scoring criteria (AASM rules).

•	Z-Score Normalization: Localized standard scaling is applied to each 30-second epoch independently to ensure amplitude invariance.

### 3. Spectral HRV Feature Engineering

For each 30-second window, Welch’s method is executed to compute the Power Spectral Density (PSD) to extract 5 robust frequency-domain features:

•	Low Frequency (LF) Power: Integrated power from 0.04 to 0.15 Hz (Reflects sympathetic activity).

•	High Frequency (HF) Power: Integrated power from 0.15 to 0.40 Hz (Reflects parasympathetic/vagal modulation).

•	LF/HF Ratio: Represents autonomic sympathetic-vagal balance.

•	Normalized LF & HF (lf_nu, hf_nu): Relative power percentages calculated against total power.

### 4. Class Imbalance & Sequence Stacking

•	SMOTE Balancing: Minority sleep stages (such as Stage 1 and REM) are synthetically oversampled using SMOTE to prevent the deep learning classifier from biasing towards dominant stages (Stage 2/Wake).

•	Temporal Sequence Stacking: Consecutive epochs are stacked into overlapping sequences of 5 windows to provide the model with essential historical sleep-transition context.

### 5. Hybrid CNN-BiLSTM Deep Learning Model

•	1D-CNN Layers: Extract spatial morphology, instantaneous fluctuations, and local localized feature maps directly from the 4Hz normalized signals.

•	Residual Connections: Prevent vanishing gradients, enabling deeper network learning.

•	Bi-LSTM: Processes the feature sequence both forward and backward in time to capture long-term temporal dependencies across macro sleep structures.

### Experimental Results
The single-lead EKG pipeline achieves robust classification mapping on independent test subjects:

 -Overall Validation Accuracy: ~63.61%.
 -Overall Test Accuracy: ~63.61% , Which is a competitive result for ECG-only classification.

## Classification Report:
<img width="783" height="646" alt="Screenshot (439)" src="https://github.com/user-attachments/assets/e851d3ac-86e5-416d-8f7b-bef60d57311f" />

## Confusion Matrix
<img width="789" height="699" alt="image" src="https://github.com/user-attachments/assets/4e5d742e-81f1-4b09-96e7-a105b6b4a0e0" />

## ROC CURVE
<img width="924" height="649" alt="Screenshot (440)" src="https://github.com/user-attachments/assets/7a3968bf-263f-419c-86f6-7feb486f99bd" />

## Refrences
- https://www.sciencedirect.com/science/article/pii/S1746809423005037
- https://arxiv.org/pdf/2412.01929









