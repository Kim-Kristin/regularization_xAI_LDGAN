from cProfile import label
import sys
from xml.sax.xmlreader import InputSource


# Append needed function/module paths
sys.path.append('./src')
sys.path.append('./src/param')
sys.path.append('./src/device')
sys.path.append('./src/dataloader')
sys.path.append('./src/plot')
sys.path.append('./LIME')
sys.path.append('./LIME/limexAI')
sys.path.append('./model')
sys.path.append('./mode/FID')
# Module - Training DCGAN
import dataloader
import device
import param
import plot
import generator
import discriminator
import penalty
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from torchvision.utils import make_grid
import os
from torch.autograd import Variable
from torchvision.utils import save_image  # Speichern von Bildern
import torch.optim as optim  # Optimierungs-Algorithmen
import time
import limexAI
from limexAI import get_explanation, explanation_hook_cifar
import FID
from FID import CalcFID
import test
from tqdm import tqdm
class train_GAN():
    def train_DCGAN(NN_Discriminator, NN_Generator, model, train_loader, random_Tensor, num_epochs, device, lr, batchsize, weights_init, start_idx=1):
        torch.cuda.empty_cache()  # leert den Cache, wenn auf der GPU gearbeitet wird

        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)

        NN_Discriminator.train()
        NN_Generator.train()

        #Lossfunction
        criterion_Gen = nn.BCELoss()
        criterion_Disc = nn.BCELoss()

        # list for save State of Training
        G_losses = []
        D_losses = []
        img_list = []
        FID_scores = []

        #Optimizer
        Gen_Opt = torch.optim.Adam(
            NN_Generator.parameters(), lr=lr, betas=(0.6, 0.999))
        Dis_Opt = torch.optim.Adam(
            NN_Discriminator.parameters(), lr=lr, betas=(0.6, 0.999))

        iters=0
        print("starting Training Vanilla DCGAN")
        # Iteration over Epochen
        for epoch in range(num_epochs):

            print("Epoch:", epoch)
            # Iteration Batches
            for i, data in tqdm(enumerate(train_loader, 0)):

                img_real, label = data

                # generate Fake-Images
                fake_img = NN_Generator(random_Tensor, GAN_param=0).to(device)

                ######################
                # Train Discriminator#
                ######################

                #Gradienten = 0
                Dis_Opt.zero_grad()

                """
                1. Train Discriminators on real Images
                """
                # real Images to Disc
                pred_real = NN_Discriminator(img_real, GAN_param=0).to(device)

                # Labeling real Images
                target_real = torch.ones(img_real.size(0), 1, device=device)
                target_real = torch.flatten(target_real)

                # Calc loss
                loss_real = criterion_Disc(pred_real, target_real)

                """
                2. Train Discriminators on fake Images
                """
                # Fake Images to Disc
                pred_fake = NN_Discriminator(fake_img.detach(), GAN_param=0).to(device)

                # Labeling fake images
                target_fake = torch.zeros(fake_img.size(0), 1, device=device)
                target_fake = torch.flatten(target_fake)

                # Loss Function - Fehler des Fake-Batch wird berechnet
                loss_fake = criterion_Disc(pred_fake, target_fake)
                loss_sum_Disc = loss_real + loss_fake

                # Backprop./ Update Weights Generators

                loss_sum_Disc.backward()
                Dis_Opt.step()

                ##################
                # Train Generator#
                ##################

                #Gradienten = 0
                Gen_Opt.zero_grad()

                # Make Pred on fake images (Try to fool Disc)
                pred_Gen = NN_Discriminator(fake_img, GAN_param=0)
                #Labeling
                target_Gen = torch.ones(batchsize, 1, device=device)
                target_Gen = torch.flatten(target_Gen)

                #calc loss
                loss_Gen = criterion_Gen(pred_Gen, target_Gen)

                # Backprop./ Update Weights Generators
                loss_Gen.backward()
                Gen_Opt.step()

                #FID
                CalcFID.trainloop(iters, epoch, num_epochs, i, NN_Generator, param.g_features, param.latent_size, img_list, device, train_loader, GAN_param=0)

            reg_model = "_Van_GAN_"

            # Save losses & plot
            D_losses.append(loss_sum_Disc.item())
            G_losses.append(loss_Gen.item())
            plot.plot_metrics(reg_model, G_losses, D_losses)

            #calc FID
            fretchet_dist =  CalcFID.calculate_fretchet(img_real,fake_img,model, device=device)
            FID_scores.append(fretchet_dist.item())
            plot.plot_FID(reg_model, FID_scores)

            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFretchet_Distance: %.4f'
                    % (epoch+1, num_epochs,
                        loss_sum_Disc.item(), loss_Gen.item(),fretchet_dist))

            Path = "./outputs/metrics/"+reg_model+"Checkpoint.pth"

            checkpoint = {
                "D_Loss": D_losses,
                "G_Loss": G_losses,
                "FID": FID_scores
            }
            FILE = "checkpoint.pth"
            torch.save(checkpoint, Path)

            # Save generate Images
            train_GAN.saves_gen_samples(
                fake_img, epoch+start_idx, random_Tensor,  dir_gen_samples = './outputs/VanillaGAN')

        #Save State Train Model
        PATH_DCGAN= "./outputs/VanillaGAN.tar"
        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict(), "FID": FID_scores},  PATH_DCGAN )

        FID_scores_test = test.test_gan(NN_Generator, model, device, random_Tensor, reg_model, GAN_param=0)
        print(FID_scores_test)

        return {"Loss_G" : loss_Gen.item(), "Loss_D" : loss_sum_Disc.item(), "FID" : FID_scores, "FID_test":FID_scores_test}

    def train_GAN_with_GP(NN_Discriminator, NN_Generator, model, train_loader, random_Tensor, num_epochs, device, lr, weights_init,start_idx=1, gradient_penalty_weight=10):
        # genannten Methoden verwenden in der Regel das gleiche Stichprobenverfahren wie GP,
        # bei dem der interpolierte Wert zwischen einer realen Stichprobe und einer generierten Stichprobe verwendet wird
        #GP
        torch.cuda.empty_cache()

        #initialize G and D with weights
        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)

        # Train Mode
        NN_Discriminator.train()
        NN_Generator.train()

        #list for save state
        G_losses_GP = []
        D_losses_GP = []
        img_list = []
        FID_scores = []

        #Optimizer
        optimizer_G = optim.Adam(NN_Generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(NN_Discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        iters=0
        print("Starting Training with GAN GP")
        for epoch in range(num_epochs):
            print("Epoch:", epoch)
            for i, data in tqdm(enumerate(train_loader)):

                real_imgs, labels = data

                # Sample random points in the latent space
                fake_img = NN_Generator(random_Tensor, GAN_param=0).to(device)
                critic_fake_pred = NN_Discriminator(fake_img, GAN_param=3).reshape(-1).to(device)
                critic_real_pred = NN_Discriminator(real_imgs, GAN_param=3).reshape(-1).to(device)

                # calc: gradient penalty & calc loss
                gp = penalty.compute_GAN_gradient_penalty(NN_Discriminator, real_imgs.data, fake_img.data, device)
                d_loss = -torch.mean(critic_real_pred) + torch.mean(critic_fake_pred ) + gradient_penalty_weight * gp

                # Gradient Disc = 0
                optimizer_D.zero_grad()

                #Backprop. / Update Weights
                d_loss.backward()
                optimizer_D.step()

                # Train the generator
                fake_imgs = NN_Generator(random_Tensor, GAN_param=0).to(device)
                fake_validity = NN_Discriminator(fake_imgs, GAN_param=3).to(device)
                #calc loss
                g_loss = -torch.mean(fake_validity)

                #Gradient Gen = 0
                optimizer_G.zero_grad()

                #Backprop. & Update Weights
                g_loss.backward()
                optimizer_G.step()

                #Calc FID
                CalcFID.trainloop(iters, epoch, num_epochs, i, NN_Generator, param.g_features, param.latent_size, img_list, device, train_loader, GAN_param=0)

            # Save loss & loss
            D_losses_GP.append(d_loss.item())
            G_losses_GP.append(g_loss.item())
            reg_model = "_GP_GAN_"
            plot.plot_metrics(reg_model, G_losses_GP, D_losses_GP)

            #calc FID & loss
            fretchet_dist =  CalcFID.calculate_fretchet(real_imgs,fake_imgs,model, device=device)
            FID_scores.append(fretchet_dist.item())
            plot.plot_FID(reg_model, FID_scores)

            #Save State Metrics
            Path = "./outputs/metrics/"+reg_model+"Checkpoint.pth"

            checkpoint = {
                "D_Loss": D_losses_GP,
                "G_Loss": G_losses_GP,
                "FID": FID_scores
            }
            FILE = "checkpoint.pth"
            torch.save(checkpoint, Path)

            # Print epoch, Loss: G und D
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFretchet_Distance: %.4f'
                    % (epoch+1, num_epochs,
                        d_loss.item(), g_loss.item(),fretchet_dist))

            # Save state generate Images
            train_GAN.saves_gen_samples(
                fake_imgs, epoch+start_idx, random_Tensor,  dir_gen_samples = './outputs/GradientPenaltyGAN')

        # Save State
        PATH_GP= "./outputs/GAN_GP.tar"
        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict()}, PATH_GP)

        FID_scores_test = test.test_gan(NN_Generator, model, device, random_Tensor, reg_model, GAN_param=0)

        return G_losses_GP, D_losses_GP , FID_scores, FID_scores_test

    def train_GAN_with_WP_Normalization(NN_Discriminator, NN_Generator, model, train_loader, random_Tensor, num_epochs, device, lr, weights_init, N_Critic,start_idx=1):
        #Weightclipping ==> Wasserstein GAN
        torch.cuda.empty_cache()

        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)

        NN_Discriminator.train()
        NN_Generator.train()

        # list Save State metrics
        G_losses = []
        D_losses = []
        img_list = []
        FID_scores = []

        # Optimzer (=RMS Prop)
        Gen_Opt = torch.optim.RMSprop(NN_Generator.parameters(),
                                lr=0.00005)
        Dis_Opt = torch.optim.RMSprop(NN_Discriminator.parameters(),
                                lr=0.00005)

        iters=0
        print("Starting Training with WGAN")
        # Iteration über die Epochen
        for epoch in range(num_epochs):

            print("Epoch:", epoch)
            # Iteration über die Bilder
            for i, data in tqdm(enumerate(train_loader,0)):


                img_real, label = data
                ######################
                # Train Discriminator#
                ######################

                # Iteration Critic (=5)
                for _ in range(N_Critic):
                    #Gradient = 0
                    Dis_Opt.zero_grad()
                    # Generate Fakeimages
                    fake_img = NN_Generator(random_Tensor, GAN_param=0).to(device)
                    # Prediction critic on real Images
                    pred_real = NN_Discriminator(img_real, GAN_param=2).to(device)
                    # Prediction critic on fake Images
                    pred_fake = NN_Discriminator(fake_img, GAN_param=2).to(device)
                    # calc Loss
                    loss_critic = -(torch.mean(pred_real)-torch.mean(pred_fake))
                    # Backprop & Update Optimzer
                    loss_critic.backward(retain_graph=True)
                    Dis_Opt.step()
                    # Weight Clipping (= 0.01) - Gesamtgradient des Diskriminators dadurch begrenzt, dass die einzelnen Gewichte im Diskriminator separat begrenzt werden.
                    for p in NN_Discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                ##################
                # Train Generator#
                ##################

                #Gradienten = 0
                Gen_Opt.zero_grad()
                fake_img = NN_Generator(random_Tensor, GAN_param=0).to(device)

                # Pred on Fakes-Images (Try to fool critic)
                pred_Gen = NN_Discriminator(fake_img, GAN_param=2).to(device)
                #calc loss
                loss_Gen = -torch.mean(pred_Gen)

                # Backprop./ Update Weights Generators
                loss_Gen.backward()
                Gen_Opt.step()

                #calc FID
                CalcFID.trainloop(iters, epoch, num_epochs, i, NN_Generator, param.g_features, param.latent_size, img_list, device, train_loader, GAN_param=0)

            # Save state of losses  Critic and Generator
            D_losses.append(loss_critic.item())
            G_losses.append(loss_Gen.item())

            #calc FID and save state
            fretchet_dist =  CalcFID.calculate_fretchet(img_real,fake_img,model, device=device) #calc FID
            FID_scores.append(fretchet_dist.item())

            # Plot results
            reg_model = "_Clipping_GAN_"
            plot.plot_metrics(reg_model, G_losses, D_losses)
            plot.plot_FID(reg_model, FID_scores)

            Path = "./outputs/metrics/"+reg_model+"Checkpoint.pth"

            # Add Metrics to a .pth-File to call the Results after the Training
            checkpoint = {
                "D_Loss": D_losses,
                "G_Loss": G_losses,
                "FID": FID_scores
            }
            FILE = "checkpoint.pth"
            torch.save(checkpoint, Path)

            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFretchet_Distance: %.4f'
                    % (epoch+1, num_epochs,
                        loss_critic.item(), loss_Gen.item(),fretchet_dist))

            # Save fake Samples/ Images
            train_GAN.saves_gen_samples(
                fake_img, epoch+start_idx, random_Tensor,  dir_gen_samples = './outputs/WeightClippingGAN')

        #Save state of the Train-Modell
        PATH_WP_Norm= "./outputs/WP_Norm.tar"
        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict()}, PATH_WP_Norm)

        FID_scores_test = test.test_gan(NN_Generator, model, device, random_Tensor, reg_model, GAN_param=0)
        return G_losses, D_losses, FID_scores, FID_scores_test

    """def train_GAN_with_WP(NN_Discriminator, NN_Generator, model, train_loader, random_Tensor, num_epochs, device, lr, batch_size, weight_init, g_features, latent_size, start_idx=1):
        # Gewichtsstrafe
        torch.cuda.empty_cache()

        NN_Discriminator.apply(weight_init)
        NN_Generator.apply(weight_init)

        NN_Discriminator.train()
        NN_Generator.train()

        G_losses = []
        D_losses = []
        img_list = []
        FID_scores = []

        D_opt = torch.optim.RMSprop(NN_Discriminator.parameters(), lr=lr)
        G_opt = torch.optim.RMSprop(NN_Generator.parameters (), lr=lr)
        D_criterion = nn.BCELoss()
        G_criterion = nn.BCELoss()

        lambda_ = 0.1
        iters=0
        #trainloop
        print("Starting Training for GAN with Weight Penalty")
        for epoch in range(0, num_epochs):
            print("Epoch: ", epoch)
            for i, data in enumerate(train_loader):
                img_real, label_real = data

                rain Discriminator
                D_opt.zero_grad()

                pred_real = NN_Discriminator(img_real, GAN_param=0).to(device)
                target_real = torch.ones(img_real.size(0),1, device=device)
                target_real = torch.flatten(target_real)

                loss_real = D_criterion(pred_real, target_real)

                #Generate fake images
                img_fake = NN_Generator(random_Tensor, GAN_param=0).to(device)

                #Training auf fake images
                pred_fake = NN_Discriminator(img_fake, GAN_param=0).to(device)
                target_fake = torch.zeros(img_fake.size(0),1, device=device)
                target_fake = torch.flatten(target_fake)

                loss_fake = D_criterion(pred_fake, target_fake)

                d_loss = loss_real + loss_fake

                # Compute the weight penalty
                weight_penalty = 0
                for param in NN_Discriminator.parameters():
                    weight_penalty += (param.norm(2) - 1) ** 2
                weight_penalty *= lambda_

                # Add the weight penalty to the discriminator loss
                d_loss += weight_penalty

                #Backpropagate and Update the Parameters
                d_loss.backward()
                D_opt.step()

                Train Generator

                G_opt.zero_grad()

                img_fake = NN_Generator(random_Tensor, GAN_param=0).to(device)

                pred_Gen = NN_Discriminator(img_fake, GAN_param=0).to(device)
                target_Gen = torch.ones(batch_size,1,device=device)
                target_Gen = torch.flatten(target_Gen)


                g_loss = G_criterion(pred_Gen, target_Gen)

                g_loss.backward()
                G_opt.step()


                CalcFID.trainloop(iters, epoch, num_epochs, i, NN_Generator, g_features, latent_size, img_list, device, train_loader, GAN_param=0)
                # Count = i #Index/ Iterationen zählen
                #print("index:", i, "D_loss:", d_loss,"G_Loss:", g_loss)

            # Speichern des Gesamtlosses von D und G und der Real und Fake Scores
            D_losses.append(d_loss.item())
            G_losses.append(g_loss.item())
            fretchet_dist =  CalcFID.calculate_fretchet(img_real, img_fake, model, device=device) #calc FID
            FID_scores.append(fretchet_dist.item())

            reg_model = "_WP_GAN_"
            plot.plot_metrics(reg_model, G_losses, D_losses)
            plot.plot_FID(reg_model, FID_scores)

            Path = "./outputs/metrics/"+reg_model+"Checkpoint.pth"

            checkpoint = {
                "D_Loss": D_losses,
                "G_Loss": G_losses,
                "FID": FID_scores
            }
            FILE = "checkpoint.pth"
            torch.save(checkpoint, Path)
            #print loss(D, G) and FID
            #if((epoch)%5 == 0):
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFretchet_Distance: %.4f'
                      % (epoch+1, num_epochs,
                         d_loss.item(), g_loss.item(),fretchet_dist))

            # Speichern der generierten Samples/ Images
            train_GAN.saves_gen_samples(
                img_fake, epoch+start_idx, random_Tensor,  dir_gen_samples = './outputs/WeightPenaltyGAN')

        PATH_DCGAN= "./outputs/WP_GAN.tar"
        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict(), "FID": FID_scores},  PATH_DCGAN )

        FID_scores_test = test.test_gan(NN_Generator, model, device, random_Tensor, reg_model)
        print(FID_scores_test)

        return {"Loss_G" : g_loss.item(), "Loss_D" : d_loss.item(), "FID" : FID_scores, "FID_test": FID_scores_test}"""

    def train_GAN_Imbalanced(NN_Discriminator, NN_Generator, model, train_loader, random_Tensor, num_epochs, device, lr, batchsize, weights_init, N_CRTIC, start_idx=1):
        # WGAN with GP


        torch.cuda.empty_cache()  # leert den Cache, wenn auf der GPU gearbeitet wird

        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)


        NN_Discriminator.train()
        NN_Generator.train()

        # Listen für Übersicht des Fortschritts
        G_losses = []
        C_losses = []
        img_list = []
        FID_scores = []

        #Optimizer
        Gen_Opt = torch.optim.Adam(
            NN_Generator.parameters(), lr=1e-4, betas=(0, 0.9))
        critic_Opt = torch.optim.Adam(
            NN_Discriminator.parameters(), lr=1e-4, betas=(0, 0.9))

        iters = 0
        LAMBDA_GP = 10  # Penalty Coefficient

        print("Starting Training with WGAN-GP")
        # Iteration Epochs
        for epoch in range(num_epochs):
            print("Epoch:", epoch)

            # Iteration Batches
            for i, data in tqdm(enumerate(train_loader, 0)):

                img_real, label = data

                # Iteration Critic (=5)
                for _ in range(N_CRTIC):

                    # Train Critic

                    # Generate fake images
                    fake_img = NN_Generator(random_Tensor, GAN_param=0).to(device)
                    #Prediction on fake an real Data
                    critic_fake_pred = NN_Discriminator(fake_img, GAN_param=3).reshape(-1).to(device)
                    critic_real_pred = NN_Discriminator(img_real, GAN_param=3).reshape(-1).to(device)

                    # Calc: gradient penalty on real and fake images
                    gp = penalty.compute_gradient_penalty(NN_Discriminator, img_real.data, fake_img.data, device)
                    #calc critic loss
                    critic_loss = -torch.mean(critic_real_pred) + torch.mean(critic_fake_pred ) + LAMBDA_GP * gp

                    # Gradient = 0
                    critic_Opt.zero_grad()

                    # Backprop. + draw dyn. Graph
                    critic_loss.backward(retain_graph=True)

                    # Update Optimizer
                    critic_Opt.step()

                # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]

                # Prediction on fake images
                gen_fake = NN_Discriminator(fake_img, GAN_param=3).reshape(-1)
                #calc Generator loss
                gen_loss = -torch.mean(gen_fake)
                # Gradient = 0
                Gen_Opt.zero_grad()
                # Backprop.
                gen_loss.backward()
                # Update Weights
                Gen_Opt.step()
                # calc FID
                CalcFID.trainloop(iters, epoch, num_epochs, i, NN_Generator, param.g_features, param.latent_size, img_list, device, train_loader, GAN_param=0)

            # Save losses of Critic and Generator
            C_losses.append(critic_loss.item())
            G_losses.append(gen_loss.item())

            # Calc FID and Save state
            fretchet_dist =  CalcFID.calculate_fretchet(img_real, fake_img, model, device=device) #calc FID
            FID_scores.append(fretchet_dist.item())

            # Plot Metrics
            reg_model = "_IT_GAN_"
            plot.plot_metrics(reg_model, G_losses, C_losses)
            plot.plot_FID(reg_model, FID_scores)

            Path = "./outputs/metrics/"+reg_model+"Checkpoint.pth"

            # Add Metrics to a .pth-File to call the Results after the Training
            checkpoint = {
                "D_Loss": C_losses,
                "G_Loss": G_losses,
                "FID": FID_scores
            }
            FILE = "checkpoint.pth"
            torch.save(checkpoint, Path)

            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFretchet_Distance: %.4f'
                      % (epoch+1, num_epochs,
                         critic_loss.item(), gen_loss.item(),fretchet_dist))


            # Save fake images
            train_GAN.saves_gen_samples(fake_img, epoch+start_idx, random_Tensor,  dir_gen_samples = './outputs/WGANGP')

        # Save Trained Models
        PATH_WGAN= "./model/WGAN_GP.tar"

        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict()}, PATH_WGAN)
        FID_scores_test = test.test_gan(NN_Generator, model, device, random_Tensor, reg_model, GAN_param=0)

        return G_losses, C_losses, FID_scores, FID_scores_test

    def train_GAN_Normalization(NN_Generator, NN_Discriminator, model, train_loader, weights_init, lr, num_epochs, batchsize, random_Tensor, device, start_idx=1):
        # instancenorm

        torch.cuda.empty_cache()

        # Weight Initialization
        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)

        # Train Mode
        NN_Discriminator.train()
        NN_Generator.train()

        # Losses for Models
        criterion_Disc = nn.BCELoss()
        criterion_Gen = nn.BCELoss()

        # List for save states
        G_losses = []
        D_losses = []
        img_list = []
        FID_scores = []

        # Optimizer
        Gen_Opt = torch.optim.Adam(
            NN_Generator.parameters(), lr=lr, betas=(0.6, 0.999))
        Dis_Opt = torch.optim.Adam(
            NN_Discriminator.parameters(), lr=lr, betas=(0.6, 0.999))

        iters=0
        print("Starting Training with Instance Normalization")
        # Iteration Epoch
        for epoch in range(0, num_epochs):

            print("Epoch:", epoch)
            # Iteration Batches
            for i, data in tqdm(enumerate(train_loader, 0)):

                img_real, label = data

                # Generate Fake-Images
                fake_img = NN_Generator(random_Tensor, GAN_param=1).to(device)

                ######################
                # Train Discriminator#
                ######################

                #Gradienten = 0
                Dis_Opt.zero_grad()

                """
                1. Train Discriminator on real images
                """
                # Pred on real images
                pred_real = NN_Discriminator(img_real, GAN_param=1).to(device)
                #pred_real = torch.flatten(pred_real)

                # Labeling real images with 1
                target_real = torch.ones(img_real.size(0), 1).to(device)
                target_real = torch.flatten(target_real)

                # Calc Loss
                loss_real = criterion_Disc(pred_real, target_real)

                """
                2. Train  Discriminator on fake images
                """
                # Predicton on fake images
                pred_fake = NN_Discriminator(fake_img.detach(), GAN_param=1).to(device)

                # Labeling fake images with 0
                target_fake = torch.zeros(fake_img.size(0), 1, device=device)
                target_fake = torch.flatten(target_fake)

                # calc loss fake
                loss_fake = criterion_Disc(pred_fake, target_fake)

                # sum losses
                loss_sum_Disc = loss_real + loss_fake

                # Backprop. & Update Weights
                loss_sum_Disc.backward()
                Dis_Opt.step()

                 ##################
                # Train Generator#
                ##################

                #Gradienten = 0
                Gen_Opt.zero_grad()

                # Predicton on fake images
                pred_Gen = NN_Discriminator(fake_img, GAN_param=1)

                # Labeling
                target_Gen = torch.ones(batchsize, 1, device=device)
                target_Gen = torch.flatten(target_Gen)

                # calc loss
                loss_Gen = criterion_Gen(pred_Gen, target_Gen)

                # Backprop./ Update der Gewichte des Generators
                loss_Gen.backward()
                Gen_Opt.step()

                # calc FID
                CalcFID.trainloop(iters, epoch, num_epochs, i, NN_Generator, param.g_features, param.latent_size, img_list, device, train_loader, GAN_param=1)

            # Save State of metrics D and G
            D_losses.append(loss_sum_Disc.item())
            G_losses.append(loss_Gen.item())
            fretchet_dist =  CalcFID.calculate_fretchet(img_real,fake_img,model, device=device) #calc FID
            FID_scores.append(fretchet_dist.item())

            # plot metrics
            reg_model = "_LN_GAN_"
            plot.plot_metrics(reg_model, G_losses, D_losses)
            plot.plot_FID(reg_model, FID_scores)

            Path = "./outputs/metrics/"+reg_model+"Checkpoint.pth"

            # Add Metrics to a .pth-File to call the Results after the Training
            checkpoint = {
                "D_Loss": D_losses,
                "G_Loss": G_losses,
                "FID": FID_scores
            }
            FILE = "checkpoint.pth"
            torch.save(checkpoint, Path)

            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFretchet_Distance: %.4f'
                    % (epoch+1, num_epochs,
                        loss_sum_Disc.item(), loss_Gen.item(),fretchet_dist))

            # Save generate images
            train_GAN.saves_gen_samples(
                fake_img, epoch+start_idx, random_Tensor,  dir_gen_samples = './outputs/NormalizationGAN')

        # Save State of trained model
        PATH_DCGAN= "./outputs/GAN_Norm.tar"
        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict(), "FID": FID_scores},  PATH_DCGAN )
        FID_scores_test = test.test_gan(NN_Generator, model, device, random_Tensor, reg_model, GAN_param=1)
        print(FID_scores_test)

        return {"Loss_G" : loss_Gen.item(), "Loss_D" : loss_sum_Disc.item(), "FID" : FID_scores, "FID_test:":FID_scores_test}

    """def train_GAN_with_diffrent_Losses(NN_Discriminator, NN_Generator, model, train_loader, random_Tensor, num_epochs, device, lr, batchsize, weights_init, start_idx=1):


        # Earth-Mover-Distanz (EM) - Paper WGAN
        # Loss-Sensitive GAN (LSGAN)
        # https://github.com/ajbrock/BigGAN-PyTorch

        Hinge Loss
        torch.cuda.empty_cache()  # leert den Cache, wenn auf der GPU gearbeitet wird

        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)

        NN_Discriminator.train()
        NN_Generator.train()

        # Listen für Übersicht des Fortschritts
        G_losses = []
        D_losses = []
        img_list = []
        FID_scores = []

        criterion_R = hingeloss.HingeLoss()
        criterion_F = hingeloss.HingeLoss()

        Gen_Opt = torch.optim.Adam(
            NN_Generator.parameters(), lr=lr, betas=(0.5, 0.999))
        Dis_Opt = torch.optim.Adam(
            NN_Discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        iters=0
        k = 0
        # Iteration über die Epochen
        for epoch in range(0, num_epochs):

            print("Epoch:", epoch)
            # Iteration über die Bilder
            for i, data in tqdm(enumerate(train_loader, 0)):

                img_real, label = data
                #train_DCGAN.saves_gen_samples(img_real, epoch+start_idx, random_Tensor)
                #train_GAN.saves_gen_samples(img_real, epoch+start_idx, random_Tensor,  dir_gen_samples = './outputs/DiffrentLossesGAN')
                # Generierung von Fake-Images
                fake_img = NN_Generator(random_Tensor, GAN_param=0).to(device)

               #train_GAN.saves_gen_samples(fake_img, epoch+start_idx, random_Tensor,  dir_gen_samples = './outputs/DiffrentLossesGAN')

                ######################
                # Train Discriminator#
                ######################

                #d_loss_sum, d_loss_avg = train_DCGAN.train_discriminator(
                # NN_Discriminator, NN_Generator, img_real, Dis_Opt, random_Tensor, limited, i, batchsize, trained_data, Gen_Opt, local_explainable, device)

                #Gradienten = 0
                Dis_Opt.zero_grad()


                1. Trainieren des Diskriminators auf realen Bildern

                # Reale Bilder werden an den Diskriminator übergeben
                pred_real = NN_Discriminator(img_real, GAN_param=0).to(device)
                #pred_real = torch.flatten(pred_real)
                #print(pred_real)

                # Kennzeichnen der realen Bilder mit 1
                target_real = torch.ones(img_real.size(0), 1, device=device)
                target_real = torch.flatten(target_real)
                #print(target_real)

                # Berechnung des Losses mit realen Bildern

                #loss_real = torch.nn.ReLU()(1.0 - pred_real) #, target_real)
                #real_score = torch.mean(pred_real).item()


                2. Trainieren des Diskriminators auf den erstellten Fake_Bildern

                # Fake Bilder werden an den Diskriminator übergeben
                pred_fake = NN_Discriminator(fake_img, GAN_param=0).to(device)
                #print(pred_fake)
                # Kennzeichnen der Fake-Bilder mit 0
                target_fake = torch.zeros(fake_img.size(0), 1, device=device)
                target_fake = torch.flatten(target_fake)
                #print(target_fake)
                # Loss Function - Fehler des Fake-Batch wird berechnet
                #loss_fake = torch.nn.ReLU()(1.0 + pred_fake) #, target_fake)
                #fake_score = torch.mean(pred_fake).item()

                #loss_sum_Disc = torch.max(0.1 + pred_fake) + torch.max(0.1 - pred_real)
                #loss_sum_Disc= torch.mean(loss_sum_Disc)
                #loss_sum_Disc = loss_real + loss_fake
                #loss_sum_Disc = loss_sum_Disc.item().mean()
                hinge_loss = torch.nn.HingeEmbeddingLoss(margin=1.0, reduction='sum')
                loss_real = hinge_loss(pred_real, target_real)
                loss_fake = hinge_loss(pred_fake, target_fake)
                #loss_real = torch.mean(F.relu(1. - pred_real))
                #loss_fake = torch.mean(F.relu(1. + pred_fake))
                loss_sum_Disc = k * (torch.mean(pred_fake))-(torch.mean(pred_real))

                #loss_real = criterion_R(pred_real, target_real)
                #loss_fake = criterion_F(pred_fake, target_fake)
                #print(loss_real, loss_fake)
                #loss_sum_Disc = loss_real + loss_fake
                k = k + 0.001*((torch.mean(pred_real)) + torch.mean(pred_fake))
                loss_sum_Disc.backward()

                Dis_Opt.step()


                ##################
                # Train Generator#
                ##################

                #Gradienten = 0
                Gen_Opt.zero_grad()

                # Übergeben der Fakes-Images an den Diskriminator (Versuch den Diskriminator zu täuschen)
                pred_Gen = NN_Discriminator(fake_img, GAN_param=0).to(device)
                target_Gen = torch.ones(batchsize, 1, device=device)
                # Torch.ones gibt einen tensor zurück welcher nur den Wert 1 enthält, und dem Shape Size = BATCH_SIZE
                target_Gen = torch.flatten(target_Gen)
                pred_real = NN_Discriminator(img_real, GAN_param=0).to(device)
                #loss = F.binary_cross_entropy(pred, target)  # loss_func(pred, target)
                #criterion_Gen = nn.BCELoss()
                #loss_Gen = criterion_Gen(pred_Gen, target_Gen)
                #print("loss_Gen:", loss_Gen)
                loss_Gen = -torch.mean(pred_Gen)

                #𝑘𝑡 + 𝜆 ⋅ (𝓛𝐷(𝑥) + 𝓛𝐺)


                loss_Gen.backward()


                # Backprop./ Update der Gewichte des Generators
                Gen_Opt.step()

                CalcFID.trainloop(iters, epoch, num_epochs, i, NN_Generator, param.g_features, param.latent_size, img_list, device, train_loader, GAN_param=0)
                # Count = i #Index/ Iterationen zählen
                #print("index:", i, "D_loss:", d_loss,"G_Loss:", g_loss)

            # Speichern des Gesamtlosses von D und G und der Real und Fake Scores
            D_losses.append(loss_sum_Disc.item())
            G_losses.append(loss_Gen.item())


            fretchet_dist =  CalcFID.calculate_fretchet(img_real,fake_img,model, device=device) #calc FID
            FID_scores.append(fretchet_dist.item())

            reg_model = "_DiffLoss_GAN_"
            plot.plot_metrics(reg_model, G_losses, D_losses)
            plot.plot_FID(reg_model, FID_scores)


            Path = "./outputs/metrics/"+reg_model+"Checkpoint.pth"

            checkpoint = {
                "D_Loss": D_losses,
                "G_Loss": G_losses,
                "FID": FID_scores
            }
            FILE = "checkpoint.pth"
            torch.save(checkpoint, Path)

            #print loss(D, G) and FID
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFretchet_Distance: %.4f'
                    % (epoch+1, num_epochs,
                        loss_sum_Disc.item(), loss_Gen.item(),fretchet_dist))

            # Speichern der generierten Samples/ Images
            train_GAN.saves_gen_samples(
                fake_img, epoch+start_idx, random_Tensor,  dir_gen_samples = './outputs/DiffrentLossesGAN')

        PATH_DCGAN= "./outputs/DiffLoss_GAN.tar"
        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict(), "FID": FID_scores},  PATH_DCGAN )
        FID_scores_test = test.test_gan(NN_Generator, model, device, random_Tensor, reg_model)
        print(FID_scores_test)

        return {"Loss_G" : loss_Gen.item(), "Loss_D" : loss_sum_Disc.item(), "FID" : FID_scores, "FID_test": FID_scores_test}"""

    """def train_xAIGAN(NN_Discriminator, NN_Generator, model, weights_init, trainloader, explainable, num_epochs, random_Tensor, lr, device, start_idx=1):

        #This function runs the experiment
        #:param logging_frequency: how frequently to log each epoch (default 4)
        #:type logging_frequency: int
        #:return: None
        #:rtype: None


        print ('start experiment')

        start_time = time.time()

        #explanationSwitch = (self.epochs + 1) / 2 if self.epochs % 2 == 1 else self.epochs / 2
        explanationSwitch=0

        torch.cuda.empty_cache()  # leert den Cache, wenn auf der GPU gearbeitet wird

        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)


        NN_Discriminator.train()
        NN_Generator.train()


        num_batches = len(trainloader)
        print("num batches", num_batches)


        if explainable:
            trained_data = Variable(next(iter(trainloader))[0])
            if device == "mps" or device== "cuda":
                trained_data = trained_data.to(device)
        else:
            trained_data = None

        # track losses
        G_losses = []
        D_losses = []
        img_list = []
        FID_scores = []


        local_explainable = False

        Gen_Opt = torch.optim.Adam(
            NN_Generator.parameters(), lr=lr, betas=(0.6, 0.999))
        Dis_Opt = torch.optim.Adam(
            NN_Discriminator.parameters(), lr=lr, betas=(0.6, 0.999))

        iters = 0
        # Start training
        for epoch in range(1, num_epochs + 1):

            if explainable and (epoch - 1) == explanationSwitch:
                NN_Generator.out.register_backward_hook(limexAI.explanation_hook_cifar)
            local_explainable = True

            for n_batch, data in enumerate(trainloader):

                sys.stdout.flush()
                real_batch, labels = data
                #print ("batch number ", n_batch)

                #labels_class = torch.max(labels, 1)[1]

                N = real_batch.size(0)
                #print ("N", N)

                # 1. Train Discriminator
                # Generate fake data and detach (so gradients are not calculated for generator)
                fake_img = NN_Generator(random_Tensor, GAN_param=0).detach()


                # Reset gradients
                Dis_Opt.zero_grad()

                # 1.1 Train on Real Data
                #print ("at discriminator training")
                 # Reale Bilder werden an den Diskriminator übergeben
                pred_real = NN_Discriminator(real_batch, GAN_param = 0) #.to(device)
                #pred_real = torch.flatten(pred_real)

                # Kennzeichnen der realen Bilder mit 1
                target_real = torch.ones(real_batch.size(0), 1) # device=device)
                target_real = torch.flatten(target_real)
                # print(target_real.size())

                # Berechnung des Losses mit realen Bildern
                criterion_Disc = nn.BCELoss()
                loss_real = criterion_Disc(pred_real, target_real)

                # 1.2 Train on Fake Data

                 # Fake Bilder werden an den Diskriminator übergeben
                pred_fake = NN_Discriminator(fake_img.detach(), GAN_param=0) #.to(device)

                # Kennzeichnen der Fake-Bilder mit 0
                target_fake = torch.zeros(fake_img.size(0), 1) # device=device)
                target_fake = torch.flatten(target_fake)
                # Loss Function - Fehler des Fake-Batch wird berechnet
                loss_fake = criterion_Disc(pred_fake, target_fake)

                # Sum up error and backpropagate
                loss_sum_Disc = loss_real + loss_fake

                loss_sum_Disc.backward()
                #sprint ("D BACKWARD DONE ", loss_sum_Disc.shape)

                # 1.3 Update weights with gradients
                Dis_Opt.step()



                # 2. Train Generator
                # Generate fake data
                fake_data = NN_Generator(random_Tensor, GAN_param=0) #.to(device)

                N = fake_data.size(0)

                # Reset gradients
                Gen_Opt.zero_grad()

                # Sample noise and generate fake data
                pred_Gen = NN_Discriminator(fake_data, GAN_param=0)
                #pred_Gen = self.discriminator(fake_data).view(-1)
                target_Gen = torch.ones(fake_img.size(0), 1) # device=device)
                # Torch.ones gibt einen tensor zurück welcher nur den Wert 1 enthält, und dem Shape Size = BATCH_SIZE
                target_Gen = torch.flatten(target_Gen)

                if local_explainable:
                    #generated_data, discriminator, prediction, device, trained_data=None
                    #print("local explanation true")
                    limexAI.get_explanation(generated_data=fake_data, discriminator=NN_Discriminator, prediction=pred_Gen,
                                    device=device, trained_data=trained_data)

                # Calculate error and back-propagate
                criterion_Gen = nn.BCELoss()
                loss_Gen = criterion_Gen(pred_Gen, target_Gen)
                #print("loss_Gen:", loss_Gen)
                loss_Gen.backward()

                # clip gradients to avoid exploding gradient problem
                nn.utils.clip_grad_norm_(NN_Generator.parameters(), 10)

                # update parameters
                Gen_Opt.step()
                CalcFID.trainloop(iters, epoch, num_epochs, n_batch, NN_Generator, param.g_features, param.latent_size, img_list, device, trainloader, GAN_param=0)


            # Save Losses for plotting later
            G_losses.append(loss_Gen.item())
            D_losses.append(loss_sum_Disc.item())
            fretchet_dist =  CalcFID.calculate_fretchet(real_batch,fake_img,model, device=device) #calc FID
            FID_scores.append(fretchet_dist.item())

            #print loss(D, G) and FID
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFretchet_Distance: %.4f'
                    % (epoch+1, num_epochs,
                        loss_sum_Disc.item(), loss_Gen.item(),fretchet_dist))

            # Speichern der generierten Samples/ Images
            train_GAN.saves_gen_samples(
                fake_img, epoch+start_idx, random_Tensor,  dir_gen_samples = './outputs/xAIGAN')

        PATH_xAI= "./outputs/GAN_xAI.tar"

        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict()}, PATH_xAI )

        FID_scores_test = test.test_gan(NN_Generator, model, device, random_Tensor)
        print(FID_scores_test)

        return G_losses, D_losses, FID_scores, FID_scores_test
"""

    def train_LDGAN(NN_Discriminator, NN_Generator, model, weights_init, trainloader, explainable, num_epochs, random_Tensor, lr, device, start_idx=1):

        #explanationSwitch = (self.epochs + 1) / 2 if self.epochs % 2 == 1 else self.epochs / 2
        explanationSwitch=0

        torch.cuda.empty_cache()

        # Weight Initialization
        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)

        # Train mode
        NN_Discriminator.train()
        NN_Generator.train()

        # lossfunction
        criterion_Gen = nn.BCELoss()
        criterion_Disc = nn.BCELoss()

        if explainable:
            trained_data = Variable(next(iter(trainloader))[0])
            trained_data = trained_data.to(device)

        # lists for track metrics
        G_losses = []
        D_losses = []
        img_list = []
        FID_scores = []


        local_explainable = False

        # Optimizer
        Gen_Opt = torch.optim.Adam(
            NN_Generator.parameters(), lr=lr, betas=(0.6, 0.999))
        Dis_Opt = torch.optim.Adam(
            NN_Discriminator.parameters(), lr=lr, betas=(0.6, 0.999))

        iters = 0
        print ('Starting Training with xAI-LDGAN')
        # Start training
        for epoch in range(num_epochs):

            if explainable and (epoch - 1) == explanationSwitch:
                NN_Generator.out.register_backward_hook(limexAI.explanation_hook_cifar)
            local_explainable = True

            print("Epoch: ", epoch)
            for n_batch, data in tqdm(enumerate(trainloader)):

                sys.stdout.flush() #Flushing buffer Reference: https://www.geeksforgeeks.org/python-sys-stdout-flush/
                real_batch, labels = data

                # 2. Train Generator
                # Generate fake data
                fake_data = NN_Generator(random_Tensor, GAN_param=1).to(device)

                # Gradients = 0
                Gen_Opt.zero_grad()

                # Prediction on fake Data
                pred_Gen = NN_Discriminator(fake_data, GAN_param=1).to(device)

                #Labeling
                target_Gen = torch.ones(fake_data.size(0), 1).to(device)
                target_Gen = torch.flatten(target_Gen)

                # Activate xAI (Images explained if Statement "explainable=True")
                if local_explainable:
                    print("local explanation true")
                    #call Module LIME
                    limexAI.get_explanation(generated_data=fake_data, discriminator=NN_Discriminator, prediction=pred_Gen,
                                    device=device, trained_data=trained_data)

                # Calc error and back-propagate
                loss_Gen = criterion_Gen(pred_Gen, target_Gen)
                loss_Gen.backward()

                # clip gradients to avoid exploding gradient problem
                nn.utils.clip_grad_norm_(NN_Generator.parameters(), 10)

                # update Params
                Gen_Opt.step()

                # Train Discriminator
                # Gradients = 0
                Dis_Opt.zero_grad()

                # 1.1 Train on Real Data
                # Prediction on real Data
                pred_real = NN_Discriminator(real_batch, GAN_param=1).to(device)

                # Labeling real Data
                target_real = torch.ones(real_batch.size(0), 1).to(device)
                target_real = torch.flatten(target_real)

                # Calc Loss
                loss_real = criterion_Disc(pred_real, target_real)

                # 1.2 Train on Fake Data
                # Prediction on fake images
                pred_fake = NN_Discriminator(fake_data.detach(), GAN_param=1).to(device)

                # Labeling fake images
                target_fake = torch.zeros(fake_data.size(0), 1).to(device)
                target_fake = torch.flatten(target_fake)

                # Calc loss
                loss_fake = criterion_Disc(pred_fake, target_fake)

                # Sum up losses and backpropagate
                loss_sum_Disc = loss_real + loss_fake
                # Average of Sum up
                loss_avg = loss_sum_Disc/2

                # first half of the Training
                if n_batch < (real_batch.size(dim=0)/2):
                    #Normal
                    print("Normal-Train", n_batch, "from",real_batch.size(dim=0))
                    # Backprop & Update Weights
                    loss_sum_Disc.backward()
                    Dis_Opt.step()

                # second half of the Training
                else:
                    print("Limited-Train", n_batch, "from",real_batch.size(dim=0))

                    # Thresholds for limiting Discriminator
                    eDisc = 0.5
                    eGen = 5.0

                    # Algorithmus 2
                    # if Discriminator average loss higher 0.5
                    if loss_avg.item() > eDisc:

                        torch.autograd.set_detect_anomaly(True)
                        print("Algorithmus 2 - Disc-Average loss is higher as the Threshold = 0.5")
                        # Backprop & Update Weights
                        loss_sum_Disc.clone()
                        loss_sum_Disc.backward(retain_graph=True)
                        Dis_Opt.step()

                    #Algorithmus 3
                    # if Generator loss higher 5.0
                    elif loss_Gen > eGen:
                        print("Algorithmus 3 - Gen-loss is higher as the Threshold = 5.0")
                        # Backprop & Update Weights
                        loss_sum_Disc.backward()
                        Dis_Opt.step()

                    # Pass Update of Discriminator
                    else:
                        pass
                #calc FID
                CalcFID.trainloop(iters, epoch, num_epochs, n_batch, NN_Generator, param.g_features, param.latent_size, img_list, device, trainloader, GAN_param=1)

            # Save losses
            G_losses.append(loss_Gen.item())
            D_losses.append(loss_sum_Disc.item())

            #calc and save FID
            fretchet_dist =  CalcFID.calculate_fretchet(real_batch,fake_data,model, device) #calc FID
            FID_scores.append(fretchet_dist.item())

            # Add Metrics to a .pth-File to call the Results after the Training
            reg_model = "_LDGAN_"
            plot.plot_metrics(reg_model, G_losses, D_losses)
            plot.plot_FID(reg_model, FID_scores)

            Path = "./outputs/metrics/"+reg_model+"Checkpoint.pth"

            checkpoint = {
                "D_Loss": D_losses,
                "G_Loss": G_losses,
                "FID": FID_scores
            }
            FILE = "checkpoint.pth"
            torch.save(checkpoint, Path)
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFretchet_Distance: %.4f'
                    % (epoch+1, num_epochs,
                        loss_sum_Disc.item(), loss_Gen.item(),fretchet_dist))


            # Save generated images
            train_GAN.saves_gen_samples(
                fake_data, epoch+start_idx, random_Tensor,  dir_gen_samples = './outputs/LDGAN')

        # Save state: Trained Model
        PATH_LDGAN= "./outputs/LDGAN.tar"
        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict()}, PATH_LDGAN)

        FID_scores_test = test.test_gan(NN_Generator, model, device, random_Tensor, reg_model, GAN_param=1)
        return G_losses, D_losses, FID_scores, FID_scores_test

    # Helpfunction for Tensor Normalization
    def tensor_norm(img_tensors):
        return img_tensors * param.NORM[1][0] + param.NORM[0][0]

    # Function for Image-Visualization
    def show_images(images, nmax=64):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title("Fake_Images")
        ax.imshow(make_grid(train_GAN.tensor_norm(images.detach()[:nmax]), nrow=8).permute(
            1, 2, 0).cpu())

    # Function to save Images
    def saves_gen_samples(gen_img, idx, random_Tensor, dir_gen_samples):
        fake_img_name = "gen_img-{0:0=4d}.png".format(idx)
        os.makedirs(dir_gen_samples, exist_ok=True)
        save_image(train_GAN.tensor_norm(gen_img), os.path.join(
            dir_gen_samples, fake_img_name), nrow=8)
        train_GAN.show_images(gen_img)  # Plotting Images
        print("Saved")
