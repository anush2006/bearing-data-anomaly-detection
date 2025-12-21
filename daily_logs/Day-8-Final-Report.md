# **NASA Bearing Fault Detection — Deep Learning Project (Final Summary)**

This document presents the **final consolidated summary** of a deep learning–based anomaly detection system developed using the **NASA Bearing Dataset**.  
The project implements a **fully unsupervised CNN autoencoder pipeline**, designed, trained, validated, and evaluated end-to-end, with results aligned to known physical failure behavior.

---

## **1. Problem Statement**

The objective was to detect **bearing faults from raw vibration signals** under realistic industrial constraints:

* Fault labels are scarce and noisy
* Failure progression is gradual, not instantaneous
* Window-level ground truth is unreliable
* Real-world condition monitoring systems are typically unsupervised

Given these constraints, the problem was framed as:

> **Unsupervised anomaly detection using reconstruction error**, rather than supervised classification.

---

## **2. Dataset Overview**

* Dataset: **NASA Bearing Dataset**
* Bearings: 4
* Sensors: 2 accelerometer channels per bearing → **8 channels total**
* Sampling: Files recorded every **10 minutes**
* Failure mode: Progressive degradation until failure

Each file contains raw vibration time-series data.

---

## **3. Exploratory Data Analysis & Fault Onset Determination**

EDA was conducted using statistical features:

* RMS
* Kurtosis

Key observations:
* RMS alone was insufficient for early fault detection
* Kurtosis exhibited a clear upward trend indicating degradation
* Fault onset occurred around **15,000 minutes**

Labeling decision:
* Files **before 15,000 min → Healthy**
* Files **after 15,000 min → Faulty**

Important constraint acknowledged:
> Labeling was performed at the **file level**, not the window level.  
> Consequently, not all windows labeled as faulty are expected to be anomalous.

This reflects realistic, weakly supervised industrial data.

---

## **4. Windowing Strategy**

* Window size: **2048**
* Stride: **256**

Each window has the shape:
(channels, temporal_values) → (2, 2048)


Windows are extracted per bearing:
* Bearing 1 → channels 0–1
* Bearing 2 → channels 2–3
* Bearing 3 → channels 4–5
* Bearing 4 → channels 6–7

Final data structure:


(num_windows, 2, 2048)


---

## **5. Model Architecture**

### **Architecture Chosen**
**Shared Encoder + 4 Independent Decoders**

Rationale:
* Healthy vibration dynamics are shared across bearings → shared encoder
* Degradation patterns differ per bearing → independent decoders
* Improves generalization and interpretability compared to:
  * One autoencoder per bearing
  * A single monolithic autoencoder

This architecture was selected based on engineering tradeoffs, not convenience.

---

## **6. CNN Autoencoder Design**

### **Encoder**
* 3 × Conv1D layers
* Kernel size: 3 (standard for vibration signal analysis)
* Each layer:
  * increases channel depth
  * halves temporal resolution

Flow:


→ (2, 2048)
→ (32, 1024)
→ (64, 512)
→ (128, 256)
→ Flatten
→ Fully Connected → latent vector (128)


### **Decoders**
* One decoder per bearing
* Symmetric inverse of the encoder
* Decoder selected dynamically using `bearing_id`

---

## **7. Training Strategy**

* **Training data:** Healthy windows only
* **Split:** 80% training / 20% validation
* **Loss function:** Mean Squared Error (MSE)
* **Optimizer:** Adam (learning rate = 1e-3)
* **Batch size:** 128
* **Epochs:** 5

Observation:
* Loss convergence occurred by approximately epoch 4
* Additional training provided diminishing returns

---

## **8. Validation & Threshold Determination**

Validation was performed exclusively on healthy data.

Procedure:
1. Compute reconstruction loss on validation windows
2. Calculate:
   * Mean (μ)
   * Standard deviation (σ)
3. Define anomaly threshold:


threshold = μ + 3σ


Final values:
* Mean ≈ **0.0063**
* Std ≈ **0.0002**
* Threshold ≈ **0.0071**

This statistical thresholding method is standard and defensible for anomaly detection.

---

## **9. Faulty Data Evaluation**

Faulty windows were:
* Never used during training
* Evaluated in inference-only mode
* Compared against the fixed threshold

Expectation stated upfront:
> Due to gradual degradation, not all faulty windows are expected to exceed the threshold.

---

## **10. Final Results**

### **Per-Bearing Anomaly Contribution**

| Bearing | % Batches Above Threshold | Avg Loss | Interpretation |
|-------|---------------------------|----------|----------------|
| Bearing 1 | ~6% | ~0.0066 | Mostly healthy |
| Bearing 2 | ~5% | ~0.0051 | Healthy |
| Bearing 3 | **100%** | ~0.0160 | Failed |
| Bearing 4 | **~90%** | ~0.0133 | Failed |

Overall faulty reconstruction loss ≈ **0.0103**

### **Key Outcome**
Bearings **3 and 4** dominate anomaly contribution, while bearings **1 and 2** remain largely within the healthy loss distribution.  
This aligns precisely with the known experimental failure conditions of the dataset.

---

## **11. Validation of the Pipeline**

The results validate:
* EDA-driven fault onset determination
* Healthy-only training paradigm
* Windowing strategy
* CNN-based feature extraction
* Statistical thresholding (μ + 3σ)
* Shared-encoder / multi-decoder architecture

The observed behavior is consistent with real-world bearing degradation.

---

## **12. Conclusion**

This project demonstrates a **complete, unsupervised deep learning pipeline** for bearing fault detection using raw vibration data.

The system:
* Learns healthy operational patterns
* Detects deviations without explicit fault labels
* Handles weak supervision realistically
* Produces results aligned with physical failure behavior

The project progressed from conceptual design to validated system behavior, resulting in a robust and defensible anomaly detection solution.

**Project status: Complete.** ✅

---

## **13. Context: Prior ML Work**

This deep learning work builds on prior classical ML experience, including **credit card fraud detection**, where:
* Extreme class imbalance was handled correctly
* Appropriate evaluation metrics were selected
* Multiple models (SVM, Random Forest, XGBoost) were benchmarked
* Final model selection was justified analytically

Together, these projects demonstrate progression from **classical ML fundamentals → applied deep learning → system-level ML engineering**.


ChatGPT can make mistakes. Ch