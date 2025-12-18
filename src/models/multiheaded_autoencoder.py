import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        """Shared encoder for multi-headed autoencoder. init defines the flow of data.
        Input: (batch_size, 2, 2048)"""
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)   # 2048 → 1024
        """Downsample by 2"""
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)               # 1024 → 512
        """Downsample by 2"""
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)               # 512 → 256
        """Flatten for FC layer"""
        self.flatten = nn.Flatten()
        """Fully connected layer to latent space"""
        self.fc_latent = nn.Linear(128 * 256, latent_dim)

    def forward(self, x):
        """Forward pass through shared encoder."""
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = torch.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.flatten(x)
        z = self.fc_latent(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.fc_expand = nn.Linear(latent_dim, 128 * 256)

        self.up1 = nn.Upsample(scale_factor=2)
        self.deconv1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2)
        self.deconv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)

        self.up3 = nn.Upsample(scale_factor=2)
        self.deconv3 = nn.Conv1d(32, 2, kernel_size=3, padding=1)
    def forward(self, z):
        x = self.fc_expand(z)
        x = x.view(-1, 128, 256)   # reshape to match encoder output shape

        x = self.up1(x)
        x = torch.relu(self.deconv1(x))

        x = self.up2(x)
        x = torch.relu(self.deconv2(x))

        x = self.up3(x)
        x = self.deconv3(x)   # final activation can be linear

        return x
    
class MultiDecoderAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = SharedEncoder(latent_dim=latent_dim)

        # 4 separate decoders
        self.decoders = nn.ModuleDict({
            "1": Decoder(latent_dim),
            "2": Decoder(latent_dim),
            "3": Decoder(latent_dim),
            "4": Decoder(latent_dim),
        })

    def forward(self, x, bearing_id):
        # encode input
        z = self.encoder(x)

        # choose decoder
        output_list = []
        batch_size = bearing_id.shape[0]
        for i in range(batch_size):
            decoder = self.decoders[str(int(bearing_id[i].item()))]
            reconstructed = decoder(z[i].unsqueeze(0))
            output_list.append(reconstructed)
        return torch.cat(output_list, dim=0)