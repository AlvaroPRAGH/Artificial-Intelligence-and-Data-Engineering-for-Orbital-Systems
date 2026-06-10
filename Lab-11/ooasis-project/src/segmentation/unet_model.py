import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class SmallUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Encoder
        self.encoder1 = ConvBlock(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(32, 64)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(32, 16)
        
        # Output layer
        self.output = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)
        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder con skip connections
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)
        
        return self.output(x)

def main():
    model = SmallUNet(num_classes=4)
    sample = torch.randn(1, 3, 128, 128)
    output = model(sample)
    print("=== U-Net Inspection ===")
    print(f"Input shape: {sample.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()