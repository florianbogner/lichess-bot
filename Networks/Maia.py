import torch
import torch.nn as nn
import torch.nn.functional as F
from .map_policy import make_map

class SqueezeAndExcite(nn.Module): # https://arxiv.org/abs/1709.01507
    def __init__(self, n_features, reduction=16):
        super(SqueezeAndExcite, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            SqueezeAndExcite(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(torch.add(x, self.layers(x)))
    
class ApplyPolicyMap(nn.Module):
    def __init__(self):
        super(ApplyPolicyMap, self).__init__()
        self.fc1 = torch.Tensor(make_map()).to('cuda') #5120x1858

    def forward(self, x):
        h_conv_pol_flat = torch.reshape(x, [-1, 80*8*8])
        return torch.matmul(h_conv_pol_flat, self.fc1)
    
class Maia(nn.Module):
    def __init__(self, res_blocks=6, hidden_channels=64):
        super(Maia, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=17, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        self.res_layers = []
        for i in range(res_blocks):
            self.res_layers.append(ResidualBlock(hidden_channels, hidden_channels))
        
        self.res_layers = nn.Sequential(*self.res_layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=80, kernel_size=3, padding=1),
            nn.Flatten(),
            #nn.Linear(80*8*8, 1858),
            ApplyPolicyMap()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_layers(x)
        x = self.conv2(x)
        return x
    