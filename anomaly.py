import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: compress frame
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16x16
            nn.ReLU()
        )
        # Decoder: reconstruct frame
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load or create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae_model = ConvAutoencoder().to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor()
])

ANOMALY_THRESHOLD = 0.015  # tune this on your data

def detect_anomaly(frame):
    tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed = ae_model(tensor)
        error = torch.mean((tensor - reconstructed) ** 2).item()
    is_anomaly = error > ANOMALY_THRESHOLD
    return is_anomaly, round(error, 5)

def train_autoencoder(normal_frames, epochs=20):
    """Train on normal footage so model learns what 'normal' looks like"""
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for frame in normal_frames:
            tensor = transform(frame).unsqueeze(0).to(device)
            output = ae_model(tensor)
            loss = criterion(output, tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(normal_frames):.5f}")

    torch.save(ae_model.state_dict(), "models/autoencoder.pth")
    print("Model saved!")