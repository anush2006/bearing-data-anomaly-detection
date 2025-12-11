**Day-2:**

**Agenda:**
    To implement the basic data preprocessing for 1D-CNN autoencoder style anomaly detection.

**Process:**

1. Label the data accordingly.From the EDA we observe that the Kurtosis spikes roughly after the 15000 mins mark. The total amount of minutes is around 22000 that leaves about 65-70 percent of the data as healthy / non faulty. After that point it is unhealthy/faulty.

2. Based on the labeled data extract 80% of the non faulty section of the train the CNN baseline autoencoder. The rest 20% and faulty data will be used to test the encoder.

3. Maybe try to make one AE for each bearing or just train the same AE for all the bearings and test if it generalized better. Convert the data to (numsamples , 2) in a numpy array for Pytorch usage. Convert this data to (window_size, 2) type . 

4. Trying the single encoder 4 decoders route as it seems the best route so far.

5. Higher complexity in training, basis of loss funtion cannot be identified easily. Dominant healthy patterns may hide the underlying fault patterns.

**Result:**


Using the previous day’s EDA results, we observed that:
The kurtosis begins to increase sharply around the 15,000-minute mark.
Total experiment duration is ~22,000 minutes.
Therefore, approximately 65–70% of the data corresponds to healthy operation, and the remainder can be treated as faulty.
Using this breakpoint, every file in the dataset is labeled as:

0 → healthy
1 → faulty

Initially, the goal was to implement a CNN-based Autoencoder for the NASA bearing vibration dataset. We explored two broad design options:
* Option A — Train one autoencoder per bearing (2 channels each)
* Option B — Train a single autoencoder across all 8 channels
* Option C — Shared Encoder + 4 Decoder Heads (multi-task AE)

We adopt Single Shared Encoder + 4 Independent Decoder Heads.

This gives the best balance between:
Sensitivity
Generalization
Efficiency
Interpretability

The window size is set at a fairly large 2048 samples with a stride of 256 to ensure enough detail for the CNN to learn.

Pytorch expects the data in the shape (channels, samples) hence data was converted into that form for each bearing.

Restructured the repository to make it more professional. 
