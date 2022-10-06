# Reference: https://github.com/explainable-gan/XAIGAN/blob/3110408a80ec94a15357d54ed8f3552c69a02e2e/src/models/discriminators.py

import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary


class DiscriminatorNetCifar10(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorNetCifar10, self).__init__()
        self.n_features = (3, 32, 32)
        nc, ndf = 3, 64

        self.input_layer = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 1, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 3, 2, 3, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 2, 3, 2, 3, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 3, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, 3, 1, 0, bias=False),
            nn.Dropout2d(0.4),
            nn.Softmax2d()
        )

    def forward(self, input):
        """ overrides the __call__ method of the discriminator """
        output = self.input_layer(input)
        output = self.hidden1(output)
        output = self.hidden2(output)
        output = self.hidden3(output)
        output = self.out(output)
        return output


discrimniator = DiscriminatorNetCifar10()
summary = summary(discrimniator, (3, 32, 32))
