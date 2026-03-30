# 🏃 Human Activity Recognition: From Sensors to Insights

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://human-activity-recogniser-jhmehwqkw4kk2lbnnxbeam.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
This project is an end-to-end Machine Learning pipeline that classifies human physical activities based on smartphone sensor data. While most ML projects stop at a Jupyter Notebook, this project includes a **fully deployed web dashboard** that simulates a production-level inference engine.

## 🎯 The Challenge
The core difficulty in Human Activity Recognition (HAR) is the **"Signal-to-Action"** gap. Raw accelerometer and gyroscope data are noisy and high-frequency. This project demonstrates how to bridge that gap using **Digital Signal Processing (DSP)** and **Regularized Logistic Regression**.

---

## 🏗️ System Architecture
The pipeline follows a rigorous scientific approach to data transformation:

1.  **Data Acquisition:** Tri-axial $A_x, A_y, A_z$ signals recorded at $50\text{ Hz}$.
2.  **Pre-processing:** Application of a median filter and a $20\text{ Hz}$ 3rd order low-pass Butterworth filter.
3.  **Feature Engineering:** Sliding window segmentation (2.56s) to extract **561 features** across:
    * **Time Domain:** Mean, Standard Deviation, Signal Magnitude Area (SMA), etc.
    * **Frequency Domain:** Fast Fourier Transform (FFT) components to identify rhythmic gait patterns.
4.  **Inference:** A Multi-class Logistic Regression model ($L2$ Regularized) trained on the UCI HAR Dataset.

---
## 📊 Model Performance & Accuracy
The model was evaluated using a multi-metric approach. Below are the results for the **L2-Regularized Logistic Regression** model as seen in the research phase:

### 📈 Metrics Summary
| Metric | Score (L2 Model) |
| :--- | :--- |
| **Overall Accuracy** | **98.41%** |
| **Precision** | 98.41% |
| **Recall** | 98.41% |
| **F1-Score** | 98.41% |
| **AUC** | 99.03% |

### 🔍 Confusion Matrix
The confusion matrix highlights the model's ability to distinguish between similar activities. 

> **Note:** A specific challenge was found in the "Sitting" vs. "Standing" classes (Indices 1 & 2), where the model initially had ~27 False Negatives. Switching to $L2$ Regularized Logistic Regression significantly improved the separation boundaries for these static states.

<img width="787" height="695" alt="image" src="https://github.com/user-attachments/assets/daaf6ae2-4f21-4d5e-9da8-4485b4d157ba" />



---

## 🚀 Live Demo
**Access the Web App here:** [https://human-activity-recogniser-jhmehwqkw4kk2lbnnxbeam.streamlit.app/]

### How to use:
* **Step 1:** Download the `sample_data.csv` provided in the app.
* **Step 2:** Upload the file to the dashboard.
* **Step 3:** View the predicted activity distribution and classification summary.

---

## 🛠️ Technical Deep Dive
### Why Logistic Regression ($L2$)?
We selected **Logistic Regression with L2 Regularization (Ridge)** because:
* **Standard (lr)** and **Ridge (l2)** performed almost identically, outperforming **Lasso (l1)** across every metric (Accuracy, Recall, and AUC).
* **L1 (Lasso)** had slightly lower scores (~98.35%) because its penalty is more aggressive, sometimes deleting features that are necessary to distinguish between subtle movements like "Sitting" vs. "Standing."
* **L2 (Ridge)** provides the best balance, maintaining high accuracy while ensuring the model generalizes well to new, unseen sensor data.
---


## 📖 Dataset Credits
The data used in this project is the **UCI Human Activity Recognition Using Smartphones Dataset**.
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
* **Methodology:** Experiments were carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities wearing a smartphone (Samsung Galaxy S II) on the waist.
---
    
