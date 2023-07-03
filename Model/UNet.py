import torch
import torch.nn as nn


class UNet_modified(nn.Module):
    def __init__(self):
        super(UNet_modified, self).__init__()

        # Encoder
        self.encoder_conv1 = self.conv_block(1, 64)
        self.encoder_conv2 = self.conv_block(64, 128)
        self.encoder_conv3 = self.conv_block(128, 256)
        self.encoder_conv4 = self.conv_block(256, 512)

        # Decoder
        self.decoder_upconv3 = self.upconv_block(512, 256)
        self.decoder_upconv2 = self.upconv_block(256, 128)
        self.decoder_upconv1 = self.upconv_block(128, 64)

        self.decoder_conv3 = self.normal_conv_block(512, 256)
        self.decoder_conv2 = self.normal_conv_block(256, 128)
        self.decoder_conv1 = self.normal_conv_block(128, 64)

        self.final_conv1 = self.upconv_block(64, 64)
        self.final_conv2 = nn.Conv2d(64, 1, kernel_size=1)
        self.final_activation = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def normal_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        encoder1 = self.encoder_conv1(x)
        encoder2 = self.encoder_conv2(encoder1)
        encoder3 = self.encoder_conv3(encoder2)
        encoder4 = self.encoder_conv4(encoder3)

        # Decoder path with skip connections
        decoder3 = self.decoder_upconv3(encoder4)
        decoder3 = torch.cat([encoder3, decoder3], dim=1)
        decoder3 = self.decoder_conv3(decoder3)

        decoder2 = self.decoder_upconv2(decoder3)
        decoder2 = torch.cat([encoder2, decoder2], dim=1)
        decoder2 = self.decoder_conv2(decoder2)

        decoder1 = self.decoder_upconv1(decoder2)
        decoder1 = torch.cat([encoder1, decoder1], dim=1)
        decoder1 = self.decoder_conv1(decoder1)

        output = self.final_conv1(decoder1)
        output = self.final_conv2(output)
        output = self.final_activation(output)
        output = torch.squeeze(output)
        return output
