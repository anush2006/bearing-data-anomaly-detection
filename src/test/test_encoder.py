import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from src.models.multiheaded_autoencoder import MultiDecoderAutoencoder
def test_bearing_dataset(faulty_bearing_paths):
    """This is a function to import faulty data using the BearingDataset class.
    """
    from src.dataset.dataset import BearingDataset
    dataset = BearingDataset(healthy_paths=faulty_bearing_paths)
    return dataset

def test_model_with_batch_losses(model, dataLoader):
    """This is a function to validate the multi-headed autoencoder model.
    """
    loss_fn = MSELoss()
    model.eval()

    loss_stored = 0
    batch_losses_b1, batch_losses_b2, batch_losses_b3, batch_losses_b4 = [], [], [], []   # store per-batch loss 

    with torch.no_grad():#disables gradient calculation for validation to save memory and computations
        for (window, bearing_ids) in dataLoader:
            window = window.to(next(model.parameters()).device)
            bearing_ids = bearing_ids.to(next(model.parameters()).device)#makes sure data is on the same device as model prevents unecessary data transfer
            reconstructed = model(window, bearing_ids)
            loss = loss_fn(reconstructed, window)

            loss_value = loss.item()
            if bearing_ids[0].item() == 1:
                batch_losses_b1.append(loss_value)
            elif bearing_ids[0].item() == 2:
                batch_losses_b2.append(loss_value)
            elif bearing_ids[0].item() == 3:
                batch_losses_b3.append(loss_value)
            elif bearing_ids[0].item() == 4:
                batch_losses_b4.append(loss_value)
            loss_stored += loss_value

    loss_stored /= len(dataLoader)
    print(f"Validation Loss: {loss_stored:.4f}")

    return loss_stored, batch_losses_b1, batch_losses_b2, batch_losses_b3, batch_losses_b4
def main():
    faulty_bearing_paths = {
        1: "data/processed/faulty_bearing1_windows.npy",
        2: "data/processed/faulty_bearing2_windows.npy",
        3: "data/processed/faulty_bearing3_windows.npy",
        4: "data/processed/faulty_bearing4_windows.npy",
    }

    dataset = test_bearing_dataset(faulty_bearing_paths)
    print(f"Dataset length: {len(dataset)}")


    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiDecoderAutoencoder(latent_dim=128).to(device)
    model.load_state_dict(
        torch.load("src/output/multiheaded_autoencoder.pth", map_location=device)
    )
    loss_stored, batch_losses_b1, batch_losses_b2, batch_losses_b3, batch_losses_b4 = test_model_with_batch_losses(model, dataloader)
    threshold = 0.0071
    total_c=0
    count = 0
    for batch_loss in batch_losses_b1:
        if batch_loss > threshold:
            count += 1
            total_c+=1
        else:
            total_c+=1
    loss_contribution_b1 = sum(batch_losses_b1)/len(batch_losses_b1) if batch_losses_b1 else 0
    print(f"Bearing 1 - Number of batches above threshold: {count} out of {total_c}, Average Loss: {loss_contribution_b1}")
    total_c=0
    count = 0
    for batch_loss in batch_losses_b2:
        if batch_loss > threshold:
            count += 1
            total_c+=1
        else:
            total_c+=1
    loss_contribution_b2 = sum(batch_losses_b2)/len(batch_losses_b2) if batch_losses_b2 else 0
    print(f"Bearing 2 - Number of batches above threshold: {count} out of {total_c}, Average Loss: {loss_contribution_b2}")
    total_c=0
    count = 0
    for batch_loss in batch_losses_b3:
        if batch_loss > threshold:
            count += 1
            total_c+=1
        else:
            total_c+=1
    loss_contribution_b3 = sum(batch_losses_b3)/len(batch_losses_b3) if batch_losses_b3 else 0
    print(f"Bearing 3 - Number of batches above threshold: {count} out of {total_c}, Average Loss: {loss_contribution_b3}")
    total_c=0
    count = 0
    for batch_loss in batch_losses_b4:
        if batch_loss > threshold:
            count += 1
            total_c+=1
        else:
            total_c+=1
    loss_contribution_b4 = sum(batch_losses_b4)/len(batch_losses_b4) if batch_losses_b4 else 0
    print(f"Bearing 4 - Number of batches above threshold: {count} out of {total_c}, Average Loss: {loss_contribution_b4}")
    print(f"Overall Validation Loss: {loss_stored}")
if __name__ == "__main__":
    main()