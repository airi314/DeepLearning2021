import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, n_input=1, n_channel=64, kernel_size = 3, stride=1, repeat = 1):
        super().__init__()

        layers = [nn.Conv1d(n_input, n_channel, kernel_size, stride)]
        for i in range(repeat - 1):
            layers.append(nn.Conv1d(n_channel, n_channel, kernel_size, stride))
        layers += [nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class M3(nn.Module):
    def __init__(self, n_input=1, n_output=31, n_channel=256):
        super().__init__()

        self.model = nn.Sequential(
            Block(n_input, n_channel, 80, 4),
            Block(n_channel, n_channel, 3)
        )
        self.fc1 = nn.Linear(n_channel, n_output)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=31, n_channel=128):
        super().__init__()

        self.model = nn.Sequential(
            Block(n_input, n_channel, 80, 4),
            Block(n_channel, n_channel, 3),
            Block(n_channel, 2*n_channel, 3),
            Block(2* n_channel, 4*n_channel, 3)
        )
        self.fc1 = nn.Linear(4*n_channel, n_output)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class M5_lstm(nn.Module):
    def __init__(self, n_input=1, n_output=31, n_channel=128):
        super().__init__()

        self.model = nn.Sequential(
            Block(n_input, n_channel, 80, 4),
            Block(n_channel, n_channel, 3),
            Block(n_channel, 2*n_channel, 3),
            Block(2* n_channel, 4*n_channel, 3)
        )
        self.lstm = nn.LSTM(4*n_channel, 128, 1, batch_first=True)
        self.fc1 = nn.Linear(128, n_output)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.lstm(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class M11(nn.Module):
    def __init__(self, n_input=1, n_output=31, n_channel=64):
        super().__init__()

        self.model = nn.Sequential(
            Block(n_input, n_channel, 80, 4),
            Block(n_channel, n_channel, 3, 1, 2),
            Block(n_channel, 2* n_channel, 3, 1, 2),
            Block(2*n_channel, 4* n_channel, 3, 1, 3),
            Block(4*n_channel, 8* n_channel, 3, 1, 2),
            )

        self.fc1 = nn.Linear(8 * n_channel, n_output)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class M18(nn.Module):
    def __init__(self, n_input=1, n_output=31, n_channel=64):
        super().__init__()

        self.model = nn.Sequential(
            Block(n_input, n_channel, 80, 4),
            Block(n_channel, n_channel, 3, 1, 4),
            Block(n_channel, 2* n_channel, 3, 1, 4),
            Block(2*n_channel, 4* n_channel, 3, 1, 4),
            Block(4*n_channel, 8* n_channel, 3, 1, 4),
            )

        self.fc1 = nn.Linear(8 * n_channel, n_output)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)
