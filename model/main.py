#Module - Training DCGAN

import sys

#Append needed function/module paths
sys.path.append('./model')
sys.path.append('./model/train')
sys.path.append('./model/test')
sys.path.append('./model/generator')
sys.path.append('./model/discriminator')
sys.path.append('./src')
sys.path.append('./src/weightinit')
sys.path.append('./src/device')
sys.path.append('./src/dataloader')
sys.path.append('./src/param')


import train
import test
from train import train_DCGAN
from discriminator import DiscriminatorNetCifar10 as NN_Discriminator
from generator import GeneratorNetworkCIFAR10 as NN_Generator
import device
import dataloader
import param
import weightinit

NN_Generator = NN_Generator().to(device.device)
NN_Discriminator = NN_Discriminator().to(device.device)

G_losses, D_losses =train_DCGAN.training(NN_Discriminator, NN_Generator, param.limited, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial, True)

