import torch
import torch.nn as nn

from .blocks import ConvBlock, DownSampleLayer, ResBlockDown, ResBlockUp


class Encoder(nn.Module):
    def __init__(self, n_in, nf):
        super(Encoder, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(n_in, nf, kernel_size=7, stride=1, padding=3),
            nn.SiLU(inplace=True),
            ResBlockDown(nf, nf, nf * 2),  # [32, 32]
            ResBlockDown(nf * 2, nf * 2, nf * 4),  # [16, 16]
            ResBlockDown(nf * 4, nf * 4, nf * 8),  # [8, 8]
            ResBlockDown(nf * 8, nf * 8, nf * 16),  # [4, 4]
            # ConvBlock(nf, nf, nf),  # [64, 64]
            # DownSampleLayer(nf, nf * 2),  # [32, 32]
            # ConvBlock(nf * 2, nf * 2, nf * 2),  # [32, 32]
            # DownSampleLayer(nf * 2, nf * 4),
            # ConvBlock(nf * 4, nf * 4, nf * 4),  # [16, 16]
            # DownSampleLayer(nf * 4, nf * 8),
            # ConvBlock(nf * 8, nf * 8, nf * 8),  # [16, 16]
            # DownSampleLayer(nf * 8, nf * 16),
            # ConvBlock(nf * 16, nf * 16, nf * 16),  # [8, 8]
        )
        self.inter_A = nn.Sequential(
            ConvBlock(nf * 16, nf * 16, nf * 16),  # [4, 4]
            nn.Conv2d(nf * 16, nf * 16, kernel_size=1, stride=1, padding=0),
        )
        self.inter_B = nn.Sequential(
            ConvBlock(nf * 16, nf * 16, nf * 16),  # [4, 4]
            nn.Conv2d(nf * 16, nf * 16, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, mode="AB"):
        x = self.main(x)

        assert len(mode) == 2, "mode must be two characters"

        latents = []
        for m in mode:
            if m == "A":
                a = self.inter_A(x)
                latents.append(a)
            elif m == "B":
                b = self.inter_B(x)
                latents.append(b)
            else:
                assert False, "mode must be 'A' or 'B'"

        res = torch.cat(latents, dim=1)

        return res


class Decoder(nn.Module):
    def __init__(self, n_out, nf):
        super(Decoder, self).__init__()

        self.main = nn.Sequential(
            ConvBlock(nf * 32, nf * 16, nf * 16),  # [4, 4]
            ConvBlock(nf * 16, nf * 16, nf * 16),  # [4, 4]
            ConvBlock(nf * 16, nf * 16, nf * 16),  # [4, 4]
            ResBlockUp(nf * 16, nf * 8, nf * 8),  # [8, 8]
            ResBlockUp(nf * 8, nf * 4, nf * 4),  # [16, 16]
            ResBlockUp(nf * 4, nf * 2, nf * 2),  # [32, 32]
            ResBlockUp(nf * 2, nf, nf),  # [64, 64]
            # nn.Upsample(scale_factor=2),
            # ConvBlock(nf * 8, nf * 4, nf * 4),  # [16, 16]
            # nn.Upsample(scale_factor=2),  # [32, 32]
            # ConvBlock(nf * 4, nf * 2, nf * 2),  # [32, 32]
            # nn.Upsample(scale_factor=2),  # [64, 64]
            # ConvBlock(nf * 2, nf, nf),  # [64, 64]
            # nn.Upsample(scale_factor=2),  # [128, 128]
            nn.Conv2d(nf, n_out, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, n_in, nf):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(n_in, nf, kernel_size=7, stride=1, padding=3),
            nn.SiLU(inplace=True),
            ResBlockDown(nf, nf, nf * 2),
            ResBlockDown(nf * 2, nf * 2, nf * 4),
            ResBlockDown(nf * 4, nf * 4, nf * 8),
            ResBlockDown(nf * 8, nf * 8, nf * 16),
            nn.Flatten(),
            nn.Linear(nf * 16 * 4 * 4, 1),
        )

    def forward(self, x):
        return self.main(x)
