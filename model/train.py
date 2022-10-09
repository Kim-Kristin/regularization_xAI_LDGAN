# Module - Training DCGAN

import torch.optim as optim  # Optimierungs-Algorithmen
from torchvision.utils import save_image  # Speichern von Bildern
from torch.autograd import Variable
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch
import inital_weight
import discriminator
import generator
import param
import device
import dataloader
import sys

# Append needed function/module paths
sys.path.append('./src')
sys.path.append('./src/param')
sys.path.append('./src/device')
sys.path.append('./src/dataloader')

# Import custom functions

# import libaries


# Train Normal DCGAN

class train_DCGAN():
    def train_discriminator(NN_Discriminator, NN_Generator, real_images, Dis_Opt, random_Tensor, device):

        #Gradienten = 0
        Dis_Opt.zero_grad()

        """
        1 Trainieren des Diskriminators auf realen Bildern
        """
        # Reale Bilder werden an den Diskriminator übergeben
        pred_real = NN_Discriminator(real_images).to(device)
        #pred_real = torch.flatten(pred_real)

        # Kennzeichnen der realen Bilder mit 1
        target_real = torch.ones(real_images.size(0), 1, device=device)
        target_real = torch.flatten(target_real)
        # print(target_real.size())

        # Berechnung des Losses mit realen Bildern
        criterion = nn.BCELoss()
        loss_real = criterion(pred_real, target_real)
        real_score = torch.mean(pred_real).item()

        """
        2 Erstellen von Fake_Bildern
        """
        # Generierung von Fakeimages
        fake_img = NN_Generator(random_Tensor).to(device)
        # print(fake_img.size())

        """
        3 Trainieren des Diskriminators auf den erstellten Fake_Bildern
        """
        # Fake Bilder werden an den Diskriminator übergeben
        pred_fake = NN_Discriminator(fake_img).to(device)

        # Kennzeichnen der Fake-Bilder mit 0
        target_fake = torch.zeros(fake_img.size(0), 1, device=device)
        target_fake = torch.flatten(target_fake)
        # Loss Function - Fehler des Fake-Batch wird berechnet
        loss_fake = F.binary_cross_entropy(pred_fake, target_fake)
        fake_score = torch.mean(pred_fake).item()

        # Berechnung des Gesamt-Loss von realen und fake Images
        loss_sum = loss_real + loss_fake

        # Update der Gewichte des Diskriminators
        loss_sum.backward()
        Dis_Opt.step()

        #print("Training disc")
        return loss_sum.item(), real_score, fake_score

    def train_generator(NN_Discriminator, NN_Generator, Gen_Opt, batchsize, random_Tensor, device):

        #Gradienten = 0
        Gen_Opt.zero_grad()

        # Generierung von Fake-Images
        fake_img = NN_Generator(random_Tensor)

        # Übergeben der Fakes-Images an den Diskriminator (Versuch den Diskriminator zu täuschen)
        pred = NN_Discriminator(fake_img)
        target = torch.ones(batchsize, 1, device=device)
        # Torch.ones gibt einen tensor zurück welcher nur den Wert 1 enthält, und dem Shape Size = BATCH_SIZE
        target = torch.flatten(target)
        loss = F.binary_cross_entropy(pred, target)  # loss_func(pred, target)

        # Backprop./ Update der Gewichte des Generators
        loss.backward()
        Gen_Opt.step()

        #print("Training Gen")
        return loss.item(), fake_img

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
        ax.imshow(make_grid(train_DCGAN.tensor_norm(images.detach()[:nmax]), nrow=8).permute(
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
        save_image(train_DCGAN.tensor_norm(gen_img), os.path.join(
            dir_gen_samples, fake_img_name), nrow=8)
        train_DCGAN.show_images(gen_img)  # Plotten der Fake_Images
        print("Gespeichert")

    def training(NN_Discriminator, NN_Generator, train_loader, random_Tensor, num_epochs, device, lr, batchsize, start_idx=1):

        torch.cuda.empty_cache()  # leert den Cache, wenn auf der GPU gearbeitet wird

        NN_Discriminator.train()
        NN_Generator.train()

        # Listen für Übersicht des Fortschritts
        R_Score = []
        F_Score = []
        G_losses = []
        D_losses = []

        # Quelle: https://medium.com/@Biboswan98/optim-adam-vs-optim-sgd-lets-dive-in-8dbf1890fbdc
        # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        # t.optim.Adam(): Methode der stochastischen Optimierung; bringt im gegensatz zum SGD eine adaptive Lernrate mit
        # Momentum muss nicht wie beim SGD händisch definiert werden sonder Adam bringt es implizit in der Berechnung schon mit
        # Vorteile: wählt für jeden Parameter eine eigene LR. Sinnvoll bei Parametern die nur mit eine geringen frequenz
        # geupdatet werden/ die mit einer hohen Frequenz geupdatet werden (beschleunigt das Lernen in Fällen,
        # in denen die geeigneten Lernraten zwischen den Parametern variieren)

        Gen_Opt = torch.optim.Adam(
            NN_Generator.parameters(), lr=lr, betas=(0.6, 0.999))
        Dis_Opt = torch.optim.Adam(
            NN_Discriminator.parameters(), lr=lr, betas=(0.6, 0.999))

        # Iteration über die Epochen
        for epoch in range(0, num_epochs):
            print("Epoch:", epoch)
            # Iteration über die Bilder
            for i, data in enumerate(train_loader, 0):
                print("Batch:", i)
                img_real, label = data
                #train_DCGAN.saves_gen_samples(img_real, epoch+start_idx, random_Tensor)
                # print(img_real.size())
                # Trainieren des Diskrimniators
                d_loss, real_score, fake_score = train_DCGAN.train_discriminator(
                    NN_Discriminator, NN_Generator, img_real, Dis_Opt, random_Tensor, device)

                # Trainieren des Generators
                g_loss, g_img = train_DCGAN.train_generator(
                    NN_Discriminator, NN_Generator, Gen_Opt, batchsize, random_Tensor, device)

                # Count = i #Index/ Iterationen zählen
                #print("index:", i, "D_loss:", d_loss,"G_Loss:", g_loss)

            # Speichern des Gesamtlosses von D und G und der Real und Fake Scores
            D_losses.append(d_loss)
            G_losses.append(g_loss)
            R_Score.append(real_score)
            F_Score.append(fake_score)

            # Ausgabe EPOCH, Loss: G und D, Scores: Real und Fake
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch+1, num_epochs, g_loss, d_loss, real_score, fake_score))

            # Speichern der generierten Samples/ Images
            train_DCGAN.saves_gen_samples(
                g_img, epoch+start_idx, random_Tensor)

        return G_losses, D_losses, R_Score, F_Score


# Train DCGAN with State-of-the-Art-Regularizationmethod

# Train LDGAN

# Radom Tensor
