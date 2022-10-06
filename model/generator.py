# Reference: https://github.com/explainable-gan/XAIGAN/blob/3110408a80ec94a15357d54ed8f3552c69a02e2e/src/models/generators.py

import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary


class GeneratorNetworkCIFAR10(torch.nn.Module):
    def __init__(self):
        super(GeneratorNetworkCIFAR10, self).__init__()
        self.n_features = 100
        self.n_out = (3, 32, 32)
        nc, nz, ngf = 3, 100, 64

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.hidden3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        """ overrides the __call__ method of the generator """
        output = self.input_layer(input)
        output = self.hidden1(output)
        output = self.hidden2(output)
        output = self.out(output)
        return output


generator = GeneratorNetworkCIFAR10()
summary(generator, (100, 32, 32))
