import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, batch_norm=True):
        super().__init__()

        layers = [ nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class DCGAN_Discriminator(nn.Module):
    def __init__(self) -> None:
        super(DCGAN_Discriminator, self).__init__()

        self.model = nn.Sequential(
            Block(3, 64, bias=True, batch_norm=False),
            Block(64, 128),
            Block(128, 256),
            Block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.model(x), 1)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)