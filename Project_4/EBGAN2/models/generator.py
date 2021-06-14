import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride= 1, padding=0, bias=False):
        super().__init__()

        self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.block(x)
        return x

# from article

class EBGAN_Generator(nn.Module):

    def __init__(self) -> None:
        super(EBGAN_Generator, self).__init__()

        self.main = nn.Sequential(
            Block(100, 512, 4, 1, 0),
            Block(512, 256, 4, 2, 1),
            Block(256, 128, 4, 2, 1),
            Block(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)

# from second source

# class EBGAN_Generator(nn.Module):
#     def __init__(self, input_dim=3, input_size=64):
#         super(EBGAN_Generator, self).__init__()
#         self.input_dim = input_dim
#         self.input_size = input_size
#
#         self.fc = nn.Sequential(
#             nn.Linear(self.input_dim, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
#             nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
#             nn.ReLU(),
#         )
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, self.input_dim, 4, 2, 1),
#             nn.Tanh(),
#         )
#         self._initialize_weights()
#
#     def forward(self, input):
#         x = self.fc(input)
#         x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
#         x = self.deconv(x)
#
#         return x

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

def ebgan(pretrained: bool = False) -> EBGAN_Generator:

    model = EBGAN_Generator()
    # if pretrained:
    #     model.load_state_dict('weights/DCGAN_ImageNet.pth')
    return model
