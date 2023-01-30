# Module - Training DCGAN

from inceptionnetwork import InceptionV3
import inceptionnetwork
import weightinit
import param
import dataloader
import device
from generator import GeneratorNetworkCIFAR10 as NN_Generator
from discriminator import DiscriminatorNetCifar10 as NN_Discriminator
from FID import CalcFID
from train import train_GAN
import sys
# Append needed function/module paths
sys.path.append('./model')
sys.path.append('./model/train')
sys.path.append('./model/test')
sys.path.append('./model/generator')
sys.path.append('./model/discriminator')
sys.path.append('./model/FID')
sys.path.append('./src')
sys.path.append('./src/weightinit')
sys.path.append('./src/device')
sys.path.append('./src/dataloader')
sys.path.append('./src/param')


#GAN - Generator and Discriminator
NN_Generator = NN_Generator().to(device.device)
NN_Discriminator = NN_Discriminator().to(device.device)

# FID - InceptionV3
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx])
model = model.to(device.device)

# DCGAN
#G_loss, D_loss, FID = train_GAN.train_DCGAN(NN_Discriminator, NN_Generator, model , dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial)

#xG_loss, xD_loss = training.train_xAIGAN(NN_Discriminator, NN_Generator, weightinit.w_initial, dataloader.train_loader, True, param.num_epochs, param.random_Tensor, param.lr, param.device)

#ldG_loss, ldD_loss = training.train_LDGAN(NN_Discriminator, NN_Generator, weightinit.w_initial, dataloader.train_loader, True, param.num_epochs, param.random_Tensor, param.lr, param.device)

#ClipG_loss, clippD_loss = training.train_GAN_with_WP_Normalization(NN_Discriminator, NN_Generator,  dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial, param.choice_gc_gs, start_idx=1)

# Weight Penalty GAN
#G_loss_WP, D_loss_WP, FID_WP = train_GAN.train_GAN_with_WP(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial, param.g_features, param.latent_size)


'''Normalization'''
G_loss_Norm, D_loss_Norm, FID_Norm = train_GAN.train_GAN_Normalization(NN_Generator, NN_Discriminator, model, dataloader.train_loader, weightinit.w_initial, param.lr, param.num_epochs, param.batch_size, param.random_Tensor, device.device)

    #testDCGAN = FID("./state_save/DCGAN.tar", NN_Generator, dataloader.test_loader, device.device, param.random_Tensor)

    # G_losses, D_losses =train_DCGAN.training(NN_Discriminator, NN_Generator, param.limited, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial, True)
