
# # Sleep-Stage-Classification-using-ECG-based-IHR-Signals
##  Abstract
"Automated sleep stage classification is vital for accessible sleep health monitoring. This project develops a high-performance pipeline for classifying five sleep stages (Wake, N1, N2, N3, and REM) using only single-lead ECG signals. Our methodology integrates digital signal processing with a hybrid CNN-LSTM architecture, combining 1D-ResNet for spatial feature extraction and Bidirectional LSTM for temporal context modeling.

To enhance classification accuracy, we utilize a dual-domain approach, merging time-domain Instantaneous Heart Rate (IHR) with spectral frequency features (LF, HF,lf_nu, hf_nu, LF/HF ratio). Key challenges like class imbalance and signal noise are addressed through SMOTE oversampling and strategic data augmentation. Evaluated on the MESA dataset, the model achieves a competitive 63% accuracy, providing a computationally efficient and interpretable alternative to traditional multi-modal sleep staging systems."


## Performance Results
The model achieved an overall accuracy of **63%**, a competitive result for ECG-only classification.

### Classification Report:
<img width="452" height="221" alt="image" src="https://github.com/user-attachments/assets/533af57f-5cab-4cae-a529-c45e39433d57" />

## Confusion Matrix
<img width="789" height="699" alt="image" src="https://github.com/user-attachments/assets/4e5d742e-81f1-4b09-96e7-a105b6b4a0e0" />

## Test accuracy
<img width="420" height="42" alt="image" src="https://github.com/user-attachments/assets/3852fb1d-5d8b-4e35-aca9-d6c7a81e3962" />



