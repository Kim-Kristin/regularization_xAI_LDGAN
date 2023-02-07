#Import runtime libraries
import os
import sys

# Module - Training DCGAN

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
#Support function for clearing terminal output
def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


#GAN
NN_Generator = NN_Generator().to(device.device)
NN_Discriminator = NN_Discriminator().to(device.device)

# FID - InceptionV3
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx])
model = model.to(device.device)


#Project main
def project_main ():
    #GAN - Generator and Discriminator

    print("##########")
    print("Regularisierung von tiefen Neuronalen Netzen zur Bildgenerierung mittels xAI:Evaluierung der Effektivität von eXplainable LDGAN gegenüber State-of-the-Art Regularisierungsmethoden​")
    print("##########")
    print("\n")
    print("Regularisierungs-Model")
    print("0. All Models")
    print("1. Vanilla DCGAN")
    print("2. Gradient Penalty - GAN")
    print("3. Weight Normalization with Clipping GAN")
    print("4. Weight Penalty - GAN")
    print("5. Imbalanced GAN")
    print("6. Layeroutput Normalization")
    print("7. GAN with different Losses for D and G")
    print("8. xAI GAN")
    print("9. LDGAN")

    print("##########")
    print("10. Call Checkpoints (Metrics - Losses and FID) of all Models")
    print("##########")

    user_input = int(input("Regularisierungs-Model:"))
    if user_input == 0:
        cls()
        G_loss, D_loss, FID, FID_test = train_GAN.train_DCGAN(NN_Discriminator, NN_Generator, model , dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial)
        GPG_loss, GPD_loss, GP_FID, FID_test_gp = train_GAN.train_GAN_with_GP(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, weightinit.w_initial)
        ClipG_loss, ClipD_loss, ClipFID, FID_test_clip = train_GAN.train_GAN_with_WP_Normalization(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, weightinit.w_initial, param.N_CRTIC)
        #G_loss_WP, D_loss_WP, FID_WP, FID_test_WP = train_GAN.train_GAN_with_WP(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial, param.g_features, param.latent_size)
        G_loss_IT, D_loss_IT, FID_IT, FID_test_IT = train_GAN.train_GAN_Imbalanced(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial, param.N_CRTIC)
        G_loss_diffLoss, D_loss_diffLoss, FID_diffLoss, FID_test_diff = train_GAN.train_GAN_with_diffrent_Losses(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial)
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

    elif user_input == 4:
        cls()
        #Weight Penalty
        #G_loss_WP, D_loss_WP, FID_WP, FID_test_WP = train_GAN.train_GAN_with_WP(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial, param.g_features, param.latent_size)

    elif user_input==5:
        cls()
        #Imbalanced Training
        G_loss_IT, D_loss_IT, FID_IT, FID_test_IT = train_GAN.train_GAN_Imbalanced(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial, param.N_CRTIC)
    elif user_input == 6:
        cls()
        #Layer Output Normalization
        G_loss_LN, D_loss_LN, FID_LN, FID_test_LN = train_GAN.train_GAN_Normalization(NN_Generator, NN_Discriminator, model, dataloader.train_loader,  weightinit.w_initial, param.lr, param.num_epochs, param.batch_size, param.random_Tensor, device.device)

    elif user_input == 7:
        cls()
        #GAN with Diffrent Losses for D and G
        G_loss_diffLoss, D_loss_diffLoss, FID_diffLoss, FID_test_diff = train_GAN.train_GAN_with_diffrent_Losses(NN_Discriminator, NN_Generator, model, dataloader.train_loader, param.random_Tensor, param.num_epochs, device.device, param.lr, param.batch_size, weightinit.w_initial)
    elif user_input == 8:
        cls()
        # GAN with xAI
        #xG_loss, xD_loss, FID_xAI, FID_test_xAI = train_GAN.train_xAIGAN(NN_Discriminator, NN_Generator, model, weightinit.w_initial, dataloader.train_loader, True, param.num_epochs, param.random_Tensor, param.lr, param.device)

    elif user_input == 9:
        cls()
        # LDGAN
        ldG_loss, ldD_loss, FID_ldgan, FID_test_ldgan= train_GAN.train_LDGAN(NN_Discriminator, NN_Generator, model, weightinit.w_initial, dataloader.train_loader, True, param.num_epochs, param.random_Tensor, param.lr, device.device)

    elif user_input == 10:
        cls()
        print("### !!!MAKE SURE THAT TRAINING STATE IS SAVED!!! - State of Metrics ###")
        print("### Vanilla DCGAN ###")
        output = torch.load("./outputs/metrics/_Van_GAN_Checkpoint.pth")
        print(output)

        print("### Gradient Penalty ###")
        output = torch.load("./outputs/metrics/_GP_GAN_Checkpoint.pth")
        print(output)

        print("### Weight Clipping ###")
        output = torch.load("./outputs/metrics/_Clipping_GAN_Checkpoint.pth") #, map_location='cpu')
        print(output)

        #print("### Weight Penalty ###")
        #output = torch.load("./outputs/metrics/_WP_GAN_Checkpoint.pth")
        #print(output)

        print("### Imbalanced Training ###")
        output = torch.load("./outputs/metrics/_IT_GAN_Checkpoint.pth")
        print(output)

        print("### Layer Output Normalization ###")
        output = torch.load("./outputs/metrics/_LN_GAN_Checkpoint.pth")
        print(output)

        print("### Different Losses for D and G ###")
        output = torch.load("./outputs/metrics/_DiffLoss_GAN_Checkpoint.pth")
        print(output)

        print("### LDGAN ###")
        output = torch.load("./outputs/metrics/_LDGAN_Checkpoint.pth")
        print(output)

project_main()


