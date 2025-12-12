**Day-3:**

**Agenda:**
Implement the baseline model for 1-D CNN and it time permits attempt optimization/pruning.

**Process:**
* This architecture is determined with respect to standard practices. The no of convolutional pooling layers is determined as 3 as it has been found to reduce the feature set into an acceptable size while keeping the detail intact. Any higher detail is lost any lower the latent representation is at the risk of being too shallow and loosing critical details. FC maintains global features whereas CN extracts rich local features.

* The latent layer has 128 values with three convolutional layer each halving the temporal resolution (downsampling by a factor of 2) Then flatten the information to a single layer, and finally put it through to the latent space representation.

* The overall process is as follows:
    * 2048 is the window size, 2 channels of data (accelerometer readings)
    * First downsampling will half the amount of temporal data we see i.e (32,1024) *(no of channels is increased to preserve information)
    * Second downsampling does the same (64,512)
    * Final and third downsampling (128,256)
    * Flatten this to a feature space of 128*256 = 32768(latent space is a single dimensional vector (128 is the size chosen))
    * The linear data goes through a FC to form the latent vector of size 128.

* The decoding process is the same in reverse where the anomaly is determined based on the reconstruction loss(Calculated through mean square loss **loss = mean( (x - x_hat)^2 ))**.  If the loss is too high we can determine that the input signal is anomalous.

* Mean reconstruction loss should be calculated from the 20% healthy data retained(Use 10% for loss calculation and 10% for testing down the line). The mean reconstruction loss can be used to set the anomaly threshold. 

* Pytorch dataset class definition, requires only two things:
    * Length of the data i.e the number of windows we have
    * How to fetch a window from the data

* The respective functions are __len__ and __getitem__ . Additionally due to having 4 decoders we need to also return bearing id.

* For a pytorch model I need to extend nn.Module class, then I need to define the specific architecture determined in init. Run the architecture in forward. Then the whole thing can be called in main function using objects. This is the same for all implementations of DL models including CNNs, RNNS and other similar neural net based models.

* Output_length = (Input_length + 2*padding - kernel_size) + Stride in convolution . This is the mathematic formula to be followed to get consistent output length while processing.


**Report/Results:**

* The overall architecture of the system is prepared. The nn implementaions have been written. Reasoning for kernel size 3 = widely used for anomaly detection in this scale. Recommended for vibration data. 

* Padding is determined based on formula to maintain output length.

* Dataset module and shared encoder and generalized decoder have been setup.
