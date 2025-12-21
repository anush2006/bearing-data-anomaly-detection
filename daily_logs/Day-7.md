**Day-7:**

**Agenda:**
To set validation threshold and test anomaly detection.

**Process:**
Epoch 1/5
Train Loss: 0.0073
Validation Loss: 0.0071

Epoch 2/5
Train Loss: 0.0069
Validation Loss: 0.0067

Epoch 3/5
Train Loss: 0.0066
Validation Loss: 0.0065

Epoch 4/5
Train Loss: 0.0064
Validation Loss: 0.0064

Epoch 5/5
Train Loss: 0.0063
Validation Loss: 0.0063
Validation Loss: 0.0063
Mean Batch Loss: 0.0063, Std Dev: 0.0002
Anomaly Detection Threshold: 0.0071
**Results:**

* The CNN-based autoencoder was trained for 5 epochs using only healthy bearing windows.
* Training and validation losses consistently decreased across epochs, indicating stable convergence.
* The close alignment between training and validation loss confirms that the model is not overfitting and is learning generalized healthy vibration patterns.
* After training completion, validation reconstruction losses were collected on the held-out healthy dataset.
* The mean (μ) and standard deviation (σ) of the validation reconstruction loss distribution were computed.
* The anomaly detection threshold was set using the statistical rule μ + 3σ.
* The final threshold value obtained was **0.0071**, which represents the upper bound of expected healthy reconstruction error.
* This threshold will be used to classify unseen windows as anomalous or non-anomalous based on their reconstruction loss.
* With the threshold established, the model is now ready for evaluation on faulty bearing data.

**Observations:**

* Loss stabilization was observed by approximately epoch 4, validating the decision to limit training to 5 epochs.
* The low variance in validation loss indicates consistent reconstruction quality across healthy windows.
* The final model and threshold provide a reliable baseline for anomaly detection on the NASA bearing dataset.
