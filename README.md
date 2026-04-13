# sleep-stage-classification-from-BCG-signals-using-deep-learning.
# Sleep Stage Classification via Cross-Modal Transformers
### Leveraging EKG and Respiration Signals from the MESA Dataset

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![MNE](https://img.shields.io/badge/MNE--Python-0.24+-green.svg)

## 📌 Project Overview
This research-oriented project implements a **Cross-Modal Transformer** architecture to classify sleep stages (Wake, N1, N2, N3, and REM). Unlike traditional methods that rely heavily on EEG, this model focuses on **peripheral signals**:
* **EKG (Electrocardiogram):** Heart Rate Variability (HRV) analysis.
* **Thoracic Respiration:** Breathing patterns and respiratory effort.

The core innovation lies in the **Cross-modal Attention mechanism**, which identifies temporal correlations between cardiac and respiratory rhythms to enhance staging accuracy.

## 🔬 Methodology & Architecture
1. **Signal Acquisition:** Utilizing the MESA (Multi-Ethnic Study of Atherosclerosis) dataset.
2. **Preprocessing:** - Band-pass filtering (0.5 - 35 Hz).
   - Resampling to a unified frequency (100 Hz).
   - Synchronization of EDF signal files with Profusion XML annotations.
3. **Feature Extraction:** Dual-stream CNNs for localized spatial feature learning from raw signals.
4. **Transformer Encoder:** Capturing long-range temporal dependencies across 30-second epochs.
5. **Classification:** Softmax layer for 5-stage sleep classification.



## 📂 Repository Structure
```text
├── Dataset/             # (Ignored) Raw MESA EDF and XML files
├── Preprocessing/       # Signal cleaning and Epoching scripts
├── Models/              # Transformer and CNN architecture definitions
├── Utils/               # Helper functions for data loading
├── .gitignore           # Ensures large signal files are not uploaded
└── README.md            # Project documentation
