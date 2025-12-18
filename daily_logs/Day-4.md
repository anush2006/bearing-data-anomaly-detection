**Day-4**

**Agenda:** 

Finish design of training loop and validation loop and start training the model. 

**Process:**:

* Setup the dataloader using DataLoader class 
* Use the already defined dataset with __len__ and __getitem__ to sequentially get batches of data
* Batch size is set as 128 intially based on convention
* The training loop is as follows:
    * for each epoch:
        * set model to training mode
        * for batch in batches:
            * reconstruction = model(batch)
            * loss = MSE(reconstructed,original)
            * set gradients to 0
            * backpropogate through the network
            * update the weights of the model
            * add loss to overall loss
    * Loss = overall loss/ length of the dataset
* The validation is as follows:
    * set gradient calculation to off
    * for batch in validation_batches:
        * reconstruction = model(batch)
        * loss = MSE(reconstructed,original)
        * add loss to overall loss
    * validation loss = overall loss/ length of validation dataset

**Results:**

* The training loop and validation loop for the system have been designed and the overall flow of DL algorithms have been realized. 
* The general blueprint to train and validate every DL problem has been identified.
* Based upon the results of the validation and training further actions will be decided.