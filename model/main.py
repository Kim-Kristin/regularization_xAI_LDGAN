#Import runtime libraries
import os
import sys

# Module - Training DCGAN
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
sys.path.append('./src/plot')

import torch
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
import plot
import test


#Support function for clearing terminal output
def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


#GAN - Generator and Discriminator
NN_Generator = NN_Generator().to(device.device)
NN_Discriminator = NN_Discriminator().to(device.device)

# call InceptionV3 for FID Score
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx])
model = model.to(device.device)


#Project main
def project_main ():

    print("##########")
    print("Regularisierung von tiefen Neuronalen Netzen zur Bildgenerierung mittels xAI:Evaluierung der Effektivität von eXplainable LDGAN gegenüber State-of-the-Art Regularisierungsmethoden​")
    print("##########")
    print("\n")
    print("Regularisierungs-Model")
    print("0. All Models")
    print("1. Vanilla DCGAN")
    print("2. Gradient Penalty ")
    print("3. Weight Normalization with Clipping GAN (WGAN)")
    print("4. Imbalanced GAN (WGAN-GP)")
    print("5. Layer-Output Normalization (Instance Normalization)")
    print("6. xAI-LDGAN")

    print("##########")
    print("7. Call Checkpoints (Metrics - Losses and FID) of all Models")
    print("##########")

    user_input = int(input("Regularisierungs-Model:"))
    if user_input == 0:
        cls()
        G_loss, D_loss, FID, FID_test = train_GAN.train_DCGAN(NN_Discriminator, NN_Generator, model , dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial)
        GPG_loss, GPD_loss, GP_FID, FID_test_gp = train_GAN.train_GAN_with_GP(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, weightinit.w_initial)
        ClipG_loss, ClipD_loss, ClipFID, FID_test_clip = train_GAN.train_GAN_with_WP_Normalization(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, weightinit.w_initial, param.N_CRTIC)
        G_loss_IT, D_loss_IT, FID_IT, FID_test_IT = train_GAN.train_GAN_Imbalanced(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial, param.N_CRTIC)
        G_loss_LN, D_loss_LN, FID_LN, FID_test_LN = train_GAN.train_GAN_Normalization(NN_Generator, NN_Discriminator, model, dataloader.train_loader,  weightinit.w_initial, param.lr, param.num_epochs, param.batch_size, param.random_Tensor, device.device)
        ldG_loss, ldD_loss, FID_ldgan, FID_test_ldgan= train_GAN.train_LDGAN(NN_Discriminator, NN_Generator, model, weightinit.w_initial, dataloader.train_loader, True, param.num_epochs, param.random_Tensor, param.lr, device.device)

    elif user_input == 1:
        cls()
        #DCGAN
        G_loss, D_loss, FID, FID_test = train_GAN.train_DCGAN(NN_Discriminator, NN_Generator, model , dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial)

    elif user_input == 2:
        cls()
        #Gradient Penalty
        GPG_loss, GPD_loss, GP_FID, FID_test_gp = train_GAN.train_GAN_with_GP(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, weightinit.w_initial)

    elif user_input == 3:
        cls()
        #Clipping
        ClipG_loss, ClipD_loss, ClipFID, FID_test_clip = train_GAN.train_GAN_with_WP_Normalization(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, weightinit.w_initial, param.N_CRTIC)

    elif user_input==4:
        cls()
        #Imbalanced Training
        G_loss_IT, D_loss_IT, FID_IT, FID_test_IT = train_GAN.train_GAN_Imbalanced(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial, param.N_CRTIC)

    elif user_input == 5:
        cls()
        #Layer Output Normalization
        G_loss_LN, D_loss_LN, FID_LN, FID_test_LN = train_GAN.train_GAN_Normalization(NN_Generator, NN_Discriminator, model, dataloader.train_loader,  weightinit.w_initial, param.lr, param.num_epochs, param.batch_size, param.random_Tensor, device.device)

    elif user_input == 6:
        cls()
        # LDGAN
        ldG_loss, ldD_loss, FID_ldgan, FID_test_ldgan= train_GAN.train_LDGAN(NN_Discriminator, NN_Generator, model, weightinit.w_initial, dataloader.train_loader, True, param.num_epochs, param.random_Tensor, param.lr, device.device)

    elif user_input == 7:
        cls()
        print("### !!!MAKE SURE THAT TRAINING STATE IS SAVED!!! - State of Metrics ###")
        print("### Vanilla DCGAN ###")
        output = torch.load("./outputs/metrics/_Van_GAN_Checkpoint.pth")
        print(output)

        print("### Gradient Penalty ###")
        output = torch.load("./outputs/metrics/_GP_GAN_Checkpoint.pth")
        print(output)

        print("### Weight Clipping (WGAN) ###")
        output = torch.load("./outputs/metrics/_Clipping_GAN_Checkpoint.pth")
        print(output)

        print("### Imbalanced Training (WGAN-GP) ###")
        output = torch.load("./outputs/metrics/_IT_GAN_Checkpoint.pth")
        print(output)

        print("### Layer-Output Normalization (Instance Normalization) ###")
        output = torch.load("./outputs/metrics/_LN_GAN_Checkpoint.pth")
        print(output)

        print("### xAI-LDGAN ###")
        output = torch.load("./outputs/metrics/_LDGAN_Checkpoint.pth")
        print(output)

project_main()


