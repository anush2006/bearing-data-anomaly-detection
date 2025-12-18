**Day-5:**

Work was paused due to medical recovery following a wrist injury. Conceptual review of anomaly thresholding was completed.

During the day, a conceptual review of anomaly thresholding for the CNN-based autoencoder was carried out. It was clarified that anomaly detection is performed by analyzing the distribution of reconstruction losses obtained from unseen healthy validation data. Rather than using the mean loss alone, the threshold is defined using statistical properties of the loss distribution (mean and standard deviation), typically using a μ + kσ rule or percentile-based thresholding. This approach is standard in anomaly detection and provides a defensible method for separating normal and anomalous behavior.