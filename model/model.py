import torch
import torch.nn as nn

class CNN3D(nn.Module):
    def __init__(self, in_channels=1, latent_dim=256):
        super(CNN3D, self).__init__()
        # Encoder
        self.encoder_conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1)  # 64x64x25
        self.encoder_norm1 = nn.InstanceNorm3d(16)
        self.encoder_relu1 = nn.ReLU()

        self.encoder_conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)           # 32x32x13
        self.encoder_norm2 = nn.InstanceNorm3d(32)
        self.encoder_relu2 = nn.ReLU()

        self.encoder_conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)           # 16x16x7
        self.encoder_norm3 = nn.InstanceNorm3d(64)
        self.encoder_relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(64 * 10 * 10 * 7, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 64 * 10 * 10 * 7)
        self.unflatten = nn.Unflatten(1, (64, 10, 10, 7))  # Depth: 7

        self.decoder_conv1 = nn.ConvTranspose3d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 0)
        )  # Depth: 13
        self.decoder_norm1 = nn.InstanceNorm3d(32)
        self.decoder_relu1 = nn.ReLU()

        self.decoder_conv2 = nn.ConvTranspose3d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 0)
        )  # Depth: 26
        self.decoder_norm2 = nn.InstanceNorm3d(16)
        self.decoder_relu2 = nn.ReLU()

        self.decoder_conv3 = nn.ConvTranspose3d(
            16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 1)
        )  # Depth: 50
        self.decoder_sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder_relu1(self.encoder_norm1(self.encoder_conv1(x)))
        enc2 = self.encoder_relu2(self.encoder_norm2(self.encoder_conv2(enc1)))
        enc3 = self.encoder_relu3(self.encoder_norm3(self.encoder_conv3(enc2)))

        flattened = self.flatten(enc3)
        latent = self.encoder_fc(flattened)

        dec_input = self.decoder_fc(latent)
        dec_input = self.unflatten(dec_input)

        dec1 = self.decoder_relu1(self.decoder_norm1(self.decoder_conv1(dec_input)))
        dec1 = 0.5 * enc2 + 0.5 * dec1

        dec2 = self.decoder_relu2(self.decoder_norm2(self.decoder_conv2(dec1)))
        dec3 = self.decoder_sigmoid(self.decoder_conv3(dec2))

        return dec3, latent
