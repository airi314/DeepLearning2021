import torch
import torch.nn as nn

class EBGAN_Discriminator(nn.Module):
    def __init__(self, input_dim=3, input_size=64):
        super(EBGAN_Discriminator, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.ReLU(),
        )
        self.code = nn.Sequential(
            nn.Linear(64 * (self.input_size // 2) * (self.input_size // 2), 32),
            # bn and relu are excluded since code is used in pullaway_loss
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 64 * (self.input_size // 2) * (self.input_size // 2)),
            nn.BatchNorm1d(64 * (self.input_size // 2) * (self.input_size // 2)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, self.input_dim, 4, 2, 1),
            # nn.Sigmoid(),
        )
        self._initialize_weights()

    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.size()[0], -1)
        code = self.code(x)
        x = self.fc(code)
        x = x.view(-1, 64, (self.input_size // 2), (self.input_size // 2))
        x = self.deconv(x)

        return x, code

# class EBGAN_Discriminator(nn.Module):
#     def __init__(self) -> None:
#         super(EBGAN_Discriminator, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, 4, 2, 1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 4, 2, 1),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, 4, 2, 1),
#             nn.ReLU(),
#         )
#
#         self.flatten = nn.Linear(256 * 8 * 8, 32)
#
#         self.fc = nn.Sequential(
#             nn.Linear(32, 256 * 8 * 8),
#             nn.BatchNorm1d(256 * 8 * 8),
#             nn.ReLU(),
#         )
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, 2, 1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 3, 4, 2, 1),
#             nn.Sigmoid(),
#             )
#
#         self._initialize_weights()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.encoder(x)
#         x = x.view(x.size()[0], -1)
#         embedding = self.flatten(x)
#         x = self.fc(embedding)
#         x = x.view(-1, 256, 8, 8)
#         x = self.decoder(x)
#         return x, embedding

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