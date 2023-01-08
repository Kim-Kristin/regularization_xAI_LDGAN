from cProfile import label
import sys
from xml.sax.xmlreader import InputSource

# Append needed function/module paths
sys.path.append('./src')
sys.path.append('./src/param')
sys.path.append('./src/device')
sys.path.append('./src/dataloader')
sys.path.append('./LIME')
sys.path.append('./LIME/limexAI')

# Module - Training DCGAN
import dataloader
import device
import param
import generator
import discriminator
import penalty
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from torch.autograd import Variable
from torchvision.utils import save_image  # Speichern von Bildern
import torch.optim as optim  # Optimierungs-Algorithmen
import time
import limexAI
from limexAI import get_explanation, explanation_hook_cifar



class training():
    def train_DCGAN(NN_Discriminator, NN_Generator, train_loader, random_Tensor, num_epochs, device, lr, batchsize, weights_init, start_idx=1):
        torch.cuda.empty_cache()  # leert den Cache, wenn auf der GPU gearbeitet wird

        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)

        NN_Discriminator.train()
        NN_Generator.train()

        # Listen für Übersicht des Fortschritts
        #R_Score = []
        #F_Score = []
        G_losses = []
        D_losses = []


        Gen_Opt = torch.optim.Adam(
            NN_Generator.parameters(), lr=lr, betas=(0.6, 0.999))
        Dis_Opt = torch.optim.Adam(
            NN_Discriminator.parameters(), lr=lr, betas=(0.6, 0.999))

        # Iteration über die Epochen
        for epoch in range(1, num_epochs+1):

            print("Epoch:", epoch)
            # Iteration über die Bilder
            for i, data in enumerate(train_loader, 0):

                img_real, label = data
                #train_DCGAN.saves_gen_samples(img_real, epoch+start_idx, random_Tensor)

                # Generierung von Fake-Images
                fake_img = NN_Generator(random_Tensor).to(device)

                ##################
                # Train Generator#
                ##################

                #Gradienten = 0
                Gen_Opt.zero_grad()

                # Übergeben der Fakes-Images an den Diskriminator (Versuch den Diskriminator zu täuschen)
                pred_Gen = NN_Discriminator(fake_img, WGAN_param=0)
                target_Gen = torch.ones(batchsize, 1, device=device)
                # Torch.ones gibt einen tensor zurück welcher nur den Wert 1 enthält, und dem Shape Size = BATCH_SIZE
                target_Gen = torch.flatten(target_Gen)

                #loss = F.binary_cross_entropy(pred, target)  # loss_func(pred, target)
                criterion_Gen = nn.BCELoss()
                loss_Gen = criterion_Gen(pred_Gen, target_Gen)
                #print("loss_Gen:", loss_Gen)
                loss_Gen.backward()

                # Backprop./ Update der Gewichte des Generators
                Gen_Opt.step()

                ######################
                # Train Discriminator#
                ######################

                #d_loss_sum, d_loss_avg = train_DCGAN.train_discriminator(
                # NN_Discriminator, NN_Generator, img_real, Dis_Opt, random_Tensor, limited, i, batchsize, trained_data, Gen_Opt, local_explainable, device)

                #Gradienten = 0
                Dis_Opt.zero_grad()

                """
                1. Trainieren des Diskriminators auf realen Bildern
                """
                # Reale Bilder werden an den Diskriminator übergeben
                pred_real = NN_Discriminator(img_real, WGAN_param=0).to(device)
                #pred_real = torch.flatten(pred_real)

                # Kennzeichnen der realen Bilder mit 1
                target_real = torch.ones(img_real.size(0), 1, device=device)
                target_real = torch.flatten(target_real)
                # print(target_real.size())

                # Berechnung des Losses mit realen Bildern
                criterion_Disc = nn.BCELoss()
                loss_real = criterion_Disc(pred_real, target_real)
                #real_score = torch.mean(pred_real).item()

                """
                2. Trainieren des Diskriminators auf den erstellten Fake_Bildern
                """
                # Fake Bilder werden an den Diskriminator übergeben
                pred_fake = NN_Discriminator(fake_img.detach(), WGAN_param=0).to(device)

                # Kennzeichnen der Fake-Bilder mit 0
                target_fake = torch.zeros(fake_img.size(0), 1, device=device)
                target_fake = torch.flatten(target_fake)
                # Loss Function - Fehler des Fake-Batch wird berechnet
                loss_fake = criterion_Disc(pred_fake, target_fake)
                #fake_score = torch.mean(pred_fake).item()

                loss_sum_Disc = loss_real + loss_fake
                loss_sum_Disc.backward()
                Dis_Opt.step()

                # Count = i #Index/ Iterationen zählen
                #print("index:", i, "D_loss:", d_loss,"G_Loss:", g_loss)

            # Speichern des Gesamtlosses von D und G und der Real und Fake Scores
            D_losses.append(loss_sum_Disc.item())
            G_losses.append(loss_Gen.item())

            # Ausgabe EPOCH, Loss: G und D, Scores: Real und Fake
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}".format(
                epoch+1, num_epochs, loss_Gen, loss_sum_Disc))

            # Speichern der generierten Samples/ Images
            training.saves_gen_samples(
                fake_img, epoch+start_idx, random_Tensor)

        PATH_DCGAN= "./state_save/DCGAN.tar"

        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict()}, PATH_DCGAN )


        return G_losses, D_losses

    #GP-Methode
    def train_GAN_with_GP(NN_Discriminator, NN_Generator, train_loader, random_Tensor, num_epochs, device, lr, batchsize, weights_init, choice_gc_gs,start_idx=1, gradient_penalty_weight=10):
        # genannten Methoden verwenden in der Regel das gleiche Stichprobenverfahren wie GP,
        # bei dem der interpolierte Wert zwischen einer realen Stichprobe und einer generierten Stichprobe verwendet wird
        #GP
        #LGP
        #0-GP
        #Max-GP

        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)
        NN_Discriminator.train()
        NN_Generator.train()
        G_losses_GP = []
        D_losses_GP = []

        optimizer_G = optim.Adam(NN_Generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(NN_Discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(num_epochs):
            for i, (real_imgs, _) in enumerate(train_loader):
                batch_size = real_imgs.size(0).to(device)
                # Sample random points in the latent space
                random_latent_vectors = torch.randn(batch_size, random_Tensor).to(device=device)

                # Decode them to fake images
                fake_imgs = NN_Generator(random_latent_vectors)

                # Real images
                real_validity = NN_Discriminator(real_imgs).to(device)
                # Fake images
                fake_validity = NN_Discriminator(fake_imgs).to(device)

                # Train the discriminator
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                gradient_penalty = penalty.compute_gradient_penalty(NN_Discriminator, real_imgs.data, fake_imgs.data)
                d_loss += gradient_penalty_weight * gradient_penalty

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                # Train the generator
                random_latent_vectors = torch.randn(batch_size, random_Tensor).to(device)
                fake_imgs = NN_Generator(random_latent_vectors).to(device)
                fake_validity = NN_Discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                if i % 100 == 0:
                    print(f"epoch: {epoch+1}, batch: {i}, d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

            D_losses_GP.append(d_loss.item())
            G_losses_GP.append(g_loss.item())
            #R_Score.append(real_score)
            #F_Score.append(fake_score)

            # Ausgabe EPOCH, Loss: G und D, Scores: Real und Fake
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}".format(
                epoch+1, num_epochs, g_loss, d_loss))

            # Speichern der generierten Samples/ Images
            training.saves_gen_samples(
                fake_imgs, epoch+start_idx, random_Tensor)

        PATH_GP= "./state_save/GAN_GP.tar"

        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict()}, PATH_GP )

        return G_losses_GP, D_losses_GP



    def train_GAN_with_WP_Normalization(NN_Discriminator, NN_Generator, train_loader, random_Tensor, num_epochs, device, lr, batchsize, weights_init, choice_gc_gs,start_idx=1):
        # Gewichtsstrafe und Gewichtsnormalisierung


        #Gewichtsnormalisierung

        #Weightclipping ==> Wasserstein GAN
        torch.cuda.empty_cache()  # leert den Cache, wenn auf der GPU gearbeitet wird

        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)


        NN_Discriminator.train()
        NN_Generator.train()

        # Listen für Übersicht des Fortschritts
        #R_Score = []
        #F_Score = []
        G_losses = []
        D_losses = []


        Gen_Opt = torch.optim.Adam(
            NN_Generator.parameters(), lr=lr, betas=(0.6, 0.999))
        Dis_Opt = torch.optim.Adam(
            NN_Discriminator.parameters(), lr=lr, betas=(0.6, 0.999))

        # Iteration über die Epochen
        for epoch in range(1, num_epochs+1):

            print("Epoch:", epoch)
            # Iteration über die Bilder
            for i, data in enumerate(train_loader, 0):

                img_real, label = data
                #train_DCGAN.saves_gen_samples(img_real, epoch+start_idx, random_Tensor)

                # Generierung von Fake-Images
                fake_img = NN_Generator(random_Tensor).to(device)

                ##################
                # Train Generator#
                ##################

                #Gradienten = 0
                Gen_Opt.zero_grad()

                # Übergeben der Fakes-Images an den Diskriminator (Versuch den Diskriminator zu täuschen)
                pred_Gen = NN_Discriminator(fake_img, WGAN_param=0)
                target_Gen = torch.ones(batchsize, 1, device=device)
                # Torch.ones gibt einen tensor zurück welcher nur den Wert 1 enthält, und dem Shape Size = BATCH_SIZE
                target_Gen = torch.flatten(target_Gen)

                #loss = F.binary_cross_entropy(pred, target)  # loss_func(pred, target)
                criterion_Gen = nn.BCELoss()
                loss_Gen = criterion_Gen(pred_Gen, target_Gen)
                #print("loss_Gen:", loss_Gen)
                loss_Gen.backward()

                # Backprop./ Update der Gewichte des Generators
                Gen_Opt.step()

                ######################
                # Train Discriminator#
                ######################

                #d_loss_sum, d_loss_avg = train_DCGAN.train_discriminator(
                # NN_Discriminator, NN_Generator, img_real, Dis_Opt, random_Tensor, limited, i, batchsize, trained_data, Gen_Opt, local_explainable, device)

                #Gradienten = 0
                Dis_Opt.zero_grad()

                """
                1. Trainieren des Diskriminators auf realen Bildern
                """
                # Reale Bilder werden an den Diskriminator übergeben
                pred_real = NN_Discriminator(img_real, WGAN_param=0).to(device)
                #pred_real = torch.flatten(pred_real)

                # Kennzeichnen der realen Bilder mit 1
                target_real = torch.ones(img_real.size(0), 1, device=device)
                target_real = torch.flatten(target_real)
                # print(target_real.size())

                # Berechnung des Losses mit realen Bildern
                criterion_Disc = nn.BCELoss()
                loss_real = criterion_Disc(pred_real, target_real)
                #real_score = torch.mean(pred_real).item()

                """
                2. Trainieren des Diskriminators auf den erstellten Fake_Bildern
                """
                # Fake Bilder werden an den Diskriminator übergeben
                pred_fake = NN_Discriminator(fake_img.detach(), WGAN_param=0).to(device)

                # Kennzeichnen der Fake-Bilder mit 0
                target_fake = torch.zeros(fake_img.size(0), 1, device=device)
                target_fake = torch.flatten(target_fake)
                # Loss Function - Fehler des Fake-Batch wird berechnet
                loss_fake = criterion_Disc(pred_fake, target_fake)
                #fake_score = torch.mean(pred_fake).item()

                loss_sum_Disc = loss_real + loss_fake
                loss_sum_Disc.backward()
                if choice_gc_gs == 1:
                    torch.nn.utils.clip_grad_norm_(NN_Discriminator.parameters(), max_norm=2.0, norm_type=2)
                else:
                    torch.nn.utils.clip_grad_value_(NN_Discriminator.parameters(), clip_value=1.0)

                Dis_Opt.step()

                # Count = i #Index/ Iterationen zählen
                #print("index:", i, "D_loss:", d_loss,"G_Loss:", g_loss)

            # Speichern des Gesamtlosses von D und G und der Real und Fake Scores
            D_losses.append(loss_sum_Disc.item())
            G_losses.append(loss_Gen.item())
            #R_Score.append(real_score)
            #F_Score.append(fake_score)

            # Ausgabe EPOCH, Loss: G und D, Scores: Real und Fake
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}".format(
                epoch+1, num_epochs, loss_Gen, loss_sum_Disc))

            # Speichern der generierten Samples/ Images
            training.saves_gen_samples(
                fake_img, epoch+start_idx, random_Tensor)

        PATH_WP_Norm= "./state_save/WP_Norm.tar"

        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict()}, PATH_WP_Norm )
        return G_losses, D_losses

    def train_GAN_Imbalanced(NN_Discriminator, NN_Generator, train_loader, random_Tensor, num_epochs, device, lr, batchsize, weights_init, N_CRTIC, start_idx=1):
        LAMBDA_GP = 10  # Penalty Koeffizient

        # WGAN
        torch.cuda.empty_cache()  # leert den Cache, wenn auf der GPU gearbeitet wird

        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)


        NN_Discriminator.train()
        NN_Generator.train()

        # Listen für Übersicht des Fortschritts
        #R_Score = []
        #F_Score = []
        G_losses = []
        C_losses = []

        #Optimoizer
        Gen_Opt = torch.optim.RMSprop(NN_Generator.parameters(),
                              lr=lr)
        critic_Opt = torch.optim.RMSprop(NN_Discriminator.parameters(),
                              lr=lr)


        # Iteration über die Epochen
        for epoch in range(1, num_epochs+1):

            print("Epoch:", epoch)
            # Iteration über die Bilder
            for i, data in enumerate(train_loader, 0):

                img_real, label = data

                # Iteration über Anzahl Critic (=5)
                for _ in range(N_CRTIC):
                    # Trainieren des Diskrimniators
                    """
                    1. Trainieren des Diskriminators auf realen Bildern
                    """
                    fake_img = NN_Generator(random_Tensor)
                    critic_fake_pred = NN_Discriminator(fake_img).reshape(-1)
                    critic_real_pred = NN_Discriminator(img_real).reshape(-1)

                    # Berechnung: gradient penalty auf den realen and fake Images (Generiert durch Generator)
                    gp = penalty.compute_gradient_penalty(NN_Discriminator, img_real, fake_img, device)
                    critic_loss = -(torch.mean(critic_real_pred) -
                                    torch.mean(critic_fake_pred)) + LAMBDA_GP * gp

                    # Gradient = 0
                    NN_Discriminator.zero_grad()

                    # Backprop. + Aufzeichnen dynamischen Graphen
                    critic_loss.backward(retain_graph=True)

                    # Update Optimizer
                    critic_Opt.step()

                        # Trainieren des Generators: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
                gen_fake = NN_Discriminator(fake_img).reshape(-1)
                gen_loss = -torch.mean(gen_fake)

                # Gradient = 0
                NN_Generator.zero_grad()

                # Backprop.
                gen_loss.backward()

                # Update optimizer
                Gen_Opt.step()

                # Visualisierung nach Anzahl Display_Step (=500)
                display_step = 500
                cur_step = 0
                if cur_step % display_step == 0 and cur_step > 0:

                    # Ausgabe des Gen-Loss und Critic-Loss
                    print(
                        f"Step {cur_step}: Generator loss: {gen_loss}, critic loss: {critic_loss}")

                    # Speichern des Gesamtlosses von Critic/ Diskriminator und Generator
                    C_losses.append(critic_loss)
                    G_losses.append(gen_loss)


                    # Loss = 0 setzen
                    gen_loss = 0
                    critic_loss = 0

                    # Speichern der generierten Samples/ Images
                    training.saves_gen_samples(fake_img, epoch+start_idx, random_Tensor)
                cur_step += 1 # cur_step = cur_step+1

        PATH_WGAN= "./model/WGAN_GP.tar"

        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict()}, PATH_WGAN)

        return G_losses, C_losses


    def train_GAN_Normalization():

        #Batchnorm
        # layernorm
        # weightnorm
        # instancenorm
        #conditional Batchnorm
        pass

    def train_GAN_with_diffrent_Losses():

        # Earth-Mover-Distanz (EM) - Paper WGAN
        # Loss-Sensitive GAN (LSGAN)
        # https://github.com/ajbrock/BigGAN-PyTorch
        pass

    def train_xAIGAN(NN_Discriminator, NN_Generator, weights_init, trainloader, explainable, num_epochs, random_Tensor, lr, device, start_idx=1):
        """
        This function runs the experiment
        :param logging_frequency: how frequently to log each epoch (default 4)
        :type logging_frequency: int
        :return: None
        :rtype: None
        """

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


        local_explainable = False

        Gen_Opt = torch.optim.Adam(
            NN_Generator.parameters(), lr=lr, betas=(0.6, 0.999))
        Dis_Opt = torch.optim.Adam(
            NN_Discriminator.parameters(), lr=lr, betas=(0.6, 0.999))

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
                fake_img = NN_Generator(random_Tensor).detach()


                # Reset gradients
                Dis_Opt.zero_grad()

                # 1.1 Train on Real Data
                #print ("at discriminator training")
                 # Reale Bilder werden an den Diskriminator übergeben
                pred_real = NN_Discriminator(real_batch, WGAN_param = 0) #.to(device)
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
                pred_fake = NN_Discriminator(fake_img.detach(), WGAN_param=0) #.to(device)

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
                fake_data = NN_Generator(random_Tensor) #.to(device)

                N = fake_data.size(0)

                # Reset gradients
                Gen_Opt.zero_grad()

                # Sample noise and generate fake data
                pred_Gen = NN_Discriminator(fake_data, WGAN_param=0)
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

            # Save Losses for plotting later
            G_losses.append(loss_Gen.item())
            D_losses.append(loss_sum_Disc.item())

            # Ausgabe EPOCH, Loss: G und D, Scores: Real und Fake
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}".format(
                epoch+1, num_epochs, loss_Gen, loss_sum_Disc))

            # Speichern der generierten Samples/ Images
            training.saves_gen_samples(
                fake_img, epoch+start_idx, random_Tensor)

        PATH_xAI= "./state_save/GAN_xAI.tar"

        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict()}, PATH_xAI )

        return G_losses, D_losses


    def train_LDGAN(NN_Discriminator, NN_Generator, weights_init, trainloader, explainable, num_epochs, random_Tensor, lr, device, start_idx=1):
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


        local_explainable = False

        Gen_Opt = torch.optim.Adam(
            NN_Generator.parameters(), lr=lr, betas=(0.6, 0.999))
        Dis_Opt = torch.optim.Adam(
            NN_Discriminator.parameters(), lr=lr, betas=(0.6, 0.999))

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

                 # 2. Train Generator
                # Generate fake data
                fake_img = NN_Generator(random_Tensor).detach()

                fake_data = NN_Generator(random_Tensor) #.to(device)

                N = fake_data.size(0)

                # Reset gradients
                Gen_Opt.zero_grad()

                # Sample noise and generate fake data
                pred_Gen = NN_Discriminator(fake_data, WGAN_param=0)
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

                # 1. Train Discriminator
                # Generate fake data and detach (so gradients are not calculated for generator)


                # Reset gradients
                Dis_Opt.zero_grad()

                # 1.1 Train on Real Data
                #print ("at discriminator training")
                 # Reale Bilder werden an den Diskriminator übergeben
                pred_real = NN_Discriminator(real_batch, WGAN_param=0) #.to(device)
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
                pred_fake = NN_Discriminator(fake_img.detach(), WGAN_param=0) #.to(device)

                # Kennzeichnen der Fake-Bilder mit 0
                target_fake = torch.zeros(fake_img.size(0), 1) # device=device)
                target_fake = torch.flatten(target_fake)
                # Loss Function - Fehler des Fake-Batch wird berechnet
                loss_fake = criterion_Disc(pred_fake, target_fake)

                # Sum up error and backpropagate
                loss_sum_Disc = loss_real + loss_fake

                loss_avg = loss_sum_Disc/2

                if n_batch < (real_batch.size(dim=0)/2):
                    #Normal
                    print("Normal-Train", n_batch, "from",real_batch.size(dim=0)/2)
                    # Berechnung des Gesamt-Loss von realen und fake Images

                    # Update der Gewichte des Diskriminators
                    loss_sum_Disc.backward()
                    Dis_Opt.step()
                else:
                    print("Limited-Train", n_batch)
                    eDisc = 0.5
                    eGen = 5.0

                    if loss_avg.item() > eDisc:
                        #Algorithmus 2
                        torch.autograd.set_detect_anomaly(True)
                        print("Algorithmus 2")
                        loss_sum_Disc.clone()
                        loss_sum_Disc.backward(retain_graph=True)
                        Dis_Opt.step()
                    elif loss_Gen > eGen:
                        #Algorithmus 3
                        print("Algorithmus 3")

                        loss_sum_Disc.backward()
                        Dis_Opt.step()
                    else:
                        pass

            # Save Losses for plotting later
            G_losses.append(loss_Gen.item())
            D_losses.append(loss_sum_Disc.item())

            # Ausgabe EPOCH, Loss: G und D, Scores: Real und Fake
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}".format(
                epoch+1, num_epochs, loss_Gen, loss_sum_Disc))

            # Speichern der generierten Samples/ Images
            training.saves_gen_samples(
                fake_img, epoch+start_idx, random_Tensor)

        PATH_LDGAN= "./state_save/LDGAN.tar"

        torch.save({"generator": NN_Generator.state_dict(), "discriminator": NN_Discriminator.state_dict()}, PATH_LDGAN)

        return G_losses, D_losses

        # Hilfsfunktionen zur Normalisierng von Tensoren und grafischen Darstellung
    def tensor_norm(img_tensors):
        # print (img_tensors)
        # print (img_tensors * NORM [1][0] + NORM [0][0])
        return img_tensors * param.NORM[1][0] + param.NORM[0][0]

    # Anzeigen der Bilder (Grafische Darstellung)
    def show_images(images, nmax=64):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title("Fake_Images")
        ax.imshow(make_grid(training.tensor_norm(images.detach()[:nmax]), nrow=8).permute(
            1, 2, 0).cpu())  # detach() : erstellt eine neue "Ansicht",
        # sodass diese Operationen nicht mehr verfolgt werden,
        # d. h. der Gradient wird nicht berechnet und der Untergraph
        # wird nicht aufgezeichnet > Speicher wird nicht verwendet

    def saves_gen_samples(gen_img, idx, random_Tensor):
        # Randomisierter Tensor wird an den Generator übergeben
        fake_img_name = "gen_img-{0:0=4d}.png".format(idx)
        dir_gen_samples = './outputs/dir_gen_samples'
        os.makedirs('./outputs/dir_gen_samples', exist_ok=True)
        # os.makedirs(dir_gen_samples,exist_ok=True)# Setzen von Bildbezeichnungen für die Fake_Images
        # Tensor-Normalisierung; Speichern der Fake_Images im Ordner "Outputs/dir_gen_samples/"
        save_image(training.tensor_norm(gen_img), os.path.join(
            dir_gen_samples, fake_img_name), nrow=8)
        training.show_images(gen_img)  # Plotten der Fake_Images
        print("Gespeichert")
