**Day-8:**

**Agenda:**
To finish anomaly testing and validate results.

**Process:**

* The trained CNN-based autoencoder model was loaded from the saved checkpoint to avoid retraining.
* Faulty bearing window data was loaded using the same `BearingDataset` and `DataLoader` pipeline to ensure consistency with training and validation.
* The model was set to evaluation mode and gradients were disabled to perform inference only.
* Reconstruction loss was computed batch-wise for all faulty windows using Mean Squared Error (MSE).
* The previously computed anomaly detection threshold (μ + 3σ from healthy validation data) was used as a fixed decision boundary.
* Reconstruction losses were analyzed per bearing to understand individual bearing contribution to anomaly detection.
* For each bearing, the number of batches exceeding the threshold and the average reconstruction loss were calculated.

**Results:**

* The overall reconstruction loss on faulty data was significantly higher than the healthy validation loss distribution.
* Bearings 3 and 4 showed a dominant contribution to anomaly detection:
  * A majority (or all) of their batches exceeded the anomaly threshold.
  * Average reconstruction loss values were substantially higher than the threshold.
* Bearings 1 and 2 exhibited comparatively lower reconstruction losses, with only a small fraction of batches exceeding the threshold.
* This behavior aligns with the known experimental setup of the NASA bearing dataset, where bearings 3 and 4 are the primary failure points.
* The clear separation between healthy and faulty bearings validates:
  * the windowing strategy,
  * the healthy-only training paradigm,
  * the statistical thresholding approach,
  * and the multi-decoder autoencoder architecture.

* With this evaluation, the anomaly detection pipeline is complete and has been successfully validated against real failure behavior in the dataset.
