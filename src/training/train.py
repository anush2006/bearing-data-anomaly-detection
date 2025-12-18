import numpy as np
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, random_split
from src.models.multiheaded_autoencoder import MultiDecoderAutoencoder

def get_dataloader(healthy_paths, batch_size=128, transform=None):
    from src.dataset.dataset import BearingDataset

    dataset = BearingDataset(healthy_paths, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, val_dataloader


def train_model(model, dataloader, optimizer, epochs=1):
    loss_fn = MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for window, bearing_ids in dataloader:
            window = window.to(next(model.parameters()).device)
            bearing_ids = bearing_ids.to(next(model.parameters()).device)

            reconstructed = model(window, bearing_ids)
            loss = loss_fn(reconstructed, window)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f"Train Loss: {epoch_loss:.4f}")


def validate_model(model, dataloader):
    loss_fn = MSELoss()
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for window, bearing_ids in dataloader:
            window = window.to(next(model.parameters()).device)
            bearing_ids = bearing_ids.to(next(model.parameters()).device)

            reconstructed = model(window, bearing_ids)
            loss = loss_fn(reconstructed, window)

            epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    print(f"Validation Loss: {epoch_loss:.4f}")

    return epoch_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiDecoderAutoencoder( latent_dim=128).to(device)

    healthy_paths = {
        1: "data/processed/healthy_bearing1_windows.npy",
        2: "data/processed/healthy_bearing2_windows.npy",
        3: "data/processed/healthy_bearing3_windows.npy",
        4: "data/processed/healthy_bearing4_windows.npy",
    }

    train_loader, val_loader = get_dataloader(
        healthy_paths, batch_size=128
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)#lr->learning rate

    epochs = 10
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_model(model, train_loader, optimizer, epochs=1)
        validate_model(model, val_loader)


if __name__ == "__main__":
    main()
