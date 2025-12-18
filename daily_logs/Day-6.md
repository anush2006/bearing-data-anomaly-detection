**Day-5**

**Agenda:**  
Train the CNN-based autoencoder and observe training and validation behavior on healthy data.

**Process:**  
* The training and validation pipeline designed earlier was executed end-to-end.  
* The dataset was split into 80% training and 20% validation using PyTorch’s `random_split`.  
* Data was loaded using `DataLoader`, with shuffling enabled for training and disabled for validation.  
* A clarification was made regarding shuffling of temporal data:  
  * Although the raw bearing signals are temporal, the model operates on fixed-length windows.  
  * Temporal structure is preserved **within each window**, while window order does not carry semantic meaning for reconstruction.  
  * Shuffling during training helps break correlations between consecutive windows and improves gradient stability without violating temporal consistency.  
* The model was trained only on healthy windows to learn normal vibration patterns.  
* Mean Squared Error (MSE) was used as the reconstruction loss.  
* Gradients were enabled during training and explicitly disabled during validation using `torch.no_grad()`.  
* Training and validation losses were monitored per epoch to assess convergence and stability.

**Report/Results:**  
* Both training and validation reconstruction losses show a consistent decreasing trend.  
* Training and validation losses converge to similar values, indicating stable learning without overfitting.  
* The autoencoder is learning generalized healthy vibration patterns rather than memorizing individual windows.  
* Reconstruction loss values remain low and well-behaved, confirming the correctness of the preprocessing and training pipeline.  
* Based on observed convergence behavior, training is expected to stabilize within 5–10 epochs, with ~15 epochs as a conservative upper bound.  
