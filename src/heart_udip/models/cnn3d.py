import torch.nn as nn


class CNN3D(nn.Module):
    """3D convolutional autoencoder used for cine-MAE style reconstruction."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder_conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.encoder_norm1 = nn.InstanceNorm3d(16)
        self.encoder_relu1 = nn.ReLU()

        self.encoder_conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.encoder_norm2 = nn.InstanceNorm3d(32)
        self.encoder_relu2 = nn.ReLU()

        self.encoder_conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.encoder_norm3 = nn.InstanceNorm3d(64)
        self.encoder_relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(64 * 10 * 10 * 7, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 64 * 10 * 10 * 7)
        self.unflatten = nn.Unflatten(1, (64, 10, 10, 7))

        self.decoder_conv1 = nn.ConvTranspose3d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 0)
        )
        self.decoder_norm1 = nn.InstanceNorm3d(32)
        self.decoder_relu1 = nn.ReLU()

        self.decoder_conv2 = nn.ConvTranspose3d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 0)
        )
        self.decoder_norm2 = nn.InstanceNorm3d(16)
        self.decoder_relu2 = nn.ReLU()

        self.decoder_conv3 = nn.ConvTranspose3d(
            16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 1)
        )
        self.decoder_sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder_relu1(self.encoder_norm1(self.encoder_conv1(x)))
        enc2 = self.encoder_relu2(self.encoder_norm2(self.encoder_conv2(enc1)))
        enc3 = self.encoder_relu3(self.encoder_norm3(self.encoder_conv3(enc2)))

        latent = self.encoder_fc(self.flatten(enc3))

        dec_input = self.unflatten(self.decoder_fc(latent))
        dec1 = self.decoder_relu1(self.decoder_norm1(self.decoder_conv1(dec_input)))
        dec1 = 0.5 * enc2 + 0.5 * dec1
        dec2 = self.decoder_relu2(self.decoder_norm2(self.decoder_conv2(dec1)))
        reconstruction = self.decoder_sigmoid(self.decoder_conv3(dec2))

        return reconstruction, latent
