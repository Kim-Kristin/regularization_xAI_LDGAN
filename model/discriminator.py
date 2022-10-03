# Reference: https://github.com/explainable-gan/XAIGAN/blob/3110408a80ec94a15357d54ed8f3552c69a02e2e/src/models/discriminators.py

import torch
import torch.nn as nn
import numpy as np


class DiscriminatorNetwork(torch.nn.Module):
    """
    A simple three hidden-layer discriminative neural network
    """

    def __init__(self):
        super(DiscriminatorNetwork, self).__init__()
        self.n_features = (1, 32, 32)

        self.input_layer = nn.Sequential(
            nn.Linear(int(np.prod(self.n_features)), 1296),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1296, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """ overrides the __call__ method of the discriminator """
        output = input.view(input.size(0), -1)
        output = self.input_layer(output)
        output = self.hidden1(output)
        output = self.hidden2(output)
        output = self.out(output)
        return output


class DiscriminatorNetCifar10(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorNetCifar10, self).__init__()
        self.n_features = (3, 32, 32)
        nc, ndf = 3, 64

        self.input_layer = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        """ overrides the __call__ method of the discriminator """
        output = self.input_layer(input)
        output = self.hidden1(output)
        output = self.hidden2(output)
        output = self.out(output)
        return output
