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
from train import training
from discriminator import DiscriminatorNetCifar10 as NN_Discriminator
from generator import GeneratorNetworkCIFAR10 as NN_Generator
import device
import dataloader
import param
import weightinit

NN_Generator = NN_Generator().to(device.device)
NN_Discriminator = NN_Discriminator().to(device.device)

#G_loss, D_loss = training.train_DCGAN(NN_Discriminator, NN_Generator, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial)

#xG_loss, xD_loss = training.train_xAIGAN(NN_Discriminator, NN_Generator, weightinit.w_initial, dataloader.train_loader, True, param.num_epochs, param.random_Tensor, param.lr, param.device)

#ldG_loss, ldD_loss = training.train_LDGAN(NN_Discriminator, NN_Generator, weightinit.w_initial, dataloader.train_loader, True, param.num_epochs, param.random_Tensor, param.lr, param.device)

ClipG_loss, clippD_loss = training.train_GAN_with_WP_Normalization(NN_Discriminator, NN_Generator,  dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial, param.choice_gc_gs, start_idx=1)








# G_losses, D_losses =train_DCGAN.training(NN_Discriminator, NN_Generator, param.limited, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial, True)


