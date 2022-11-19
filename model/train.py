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
                pred_Gen = NN_Discriminator(fake_img)
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
                pred_real = NN_Discriminator(img_real).to(device)
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
                pred_fake = NN_Discriminator(fake_img.detach()).to(device)

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
            #R_Score.append(real_score)
            #F_Score.append(fake_score)

            # Ausgabe EPOCH, Loss: G und D, Scores: Real und Fake
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}".format(
                epoch+1, num_epochs, loss_Gen, loss_sum_Disc))

            # Speichern der generierten Samples/ Images
            training.saves_gen_samples(
                fake_img, epoch+start_idx, random_Tensor)

        PATH_Disc= "./model/Discriminator_DCGAN.tar"
        PATH_Gen= "./model/Generator_DCGAN.tar"

        torch.save(NN_Discriminator.state_dict(), PATH_Disc)
        torch.save(NN_Generator.state_dict(), PATH_Gen)

        return G_losses, D_losses


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
                pred_real = NN_Discriminator(real_batch) #.to(device)
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
                pred_fake = NN_Discriminator(fake_img.detach()) #.to(device)

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
                pred_Gen = NN_Discriminator(fake_data)
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

        PATH_Disc= "./model/Discriminator_xAIGAN.tar"
        PATH_Gen= "./model/Generator_xAIGAN.tar"

        torch.save(NN_Discriminator.state_dict(), PATH_Disc)
        torch.save(NN_Generator.state_dict(), PATH_Gen)

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
                pred_Gen = NN_Discriminator(fake_data)
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
                pred_real = NN_Discriminator(real_batch) #.to(device)
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
                pred_fake = NN_Discriminator(fake_img.detach()) #.to(device)

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

        PATH_Disc= "./model/Discriminator_LDGAN.tar"
        PATH_Gen= "./model/Generator_LDGAN.tar"

        torch.save(NN_Discriminator.state_dict(), PATH_Disc)
        torch.save(NN_Generator.state_dict(), PATH_Gen)

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

    """def train_discriminator(NN_Discriminator, NN_Generator, real_images, Dis_Opt, random_Tensor, limited, batchindx, batchsize, trained_data, Gen_Opt, local_explainable, device):


        #Gradienten = 0
        Dis_Opt.zero_grad()

        1 Trainieren des Diskriminators auf realen Bildern

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
        #real_score = torch.mean(pred_real).item()


        2 Erstellen von Fake_Bildern

        # Generierung von Fakeimages
        fake_img = NN_Generator(random_Tensor).to(device)
        # print(fake_img.size())


        3 Trainieren des Diskriminators auf den erstellten Fake_Bildern

        # Fake Bilder werden an den Diskriminator übergeben
        pred_fake = NN_Discriminator(fake_img).to(device)

        # Kennzeichnen der Fake-Bilder mit 0
        target_fake = torch.zeros(fake_img.size(0), 1, device=device)
        target_fake = torch.flatten(target_fake)
        # Loss Function - Fehler des Fake-Batch wird berechnet
        loss_fake = criterion(pred_fake, target_fake)
        #fake_score = torch.mean(pred_fake).item()

        #THRESHOLD
        loss_sum = loss_real + loss_fake
        loss_avg = loss_sum/2

        if batchindx < (real_images.size(dim=0)/2):
            #Normal
            print("Normal-Train", batchindx, "from",real_images.size(dim=0)/2)
            # Berechnung des Gesamt-Loss von realen und fake Images

            # Update der Gewichte des Diskriminators
            loss_sum.backward()
            Dis_Opt.step()
        else:
            print("Limited-Train", batchindx)
            # Berechnung des Gesamt-Loss von realen und fake Images
            eDisc = 0.5

            # Generator
            # Trainieren des Generators
            g_loss, g_img = train_DCGAN.train_generator(NN_Discriminator, NN_Generator, Gen_Opt, batchsize, random_Tensor, device, trained_data,local_explainable)
            eGen = 5.0

            if loss_avg.item() > eDisc:
                #Algorithmus 2
                torch.autograd.set_detect_anomaly(True)
                print("Algorithmus 2")
                loss_sum.clone()
                loss_sum.backward(retain_graph=True)
                Dis_Opt.step()
            elif g_loss > eGen:
                #Algorithmus 3
                print("Algorithmus 3")

                loss_sum.backward()
                Dis_Opt.step()
            else:
                pass

        #print("Training disc")
        return loss_sum.item(), loss_avg.item() #real_score, fake_score

    def train_generator(NN_Discriminator, NN_Generator, Gen_Opt, batchsize, random_Tensor, device, trained_data, local_explainable):

        #Gradienten = 0
        Gen_Opt.zero_grad()

        # Generierung von Fake-Images
        fake_img = NN_Generator(random_Tensor)

        # Übergeben der Fakes-Images an den Diskriminator (Versuch den Diskriminator zu täuschen)
        pred = NN_Discriminator(fake_img)
        target = torch.ones(batchsize, 1, device=device)
        # Torch.ones gibt einen tensor zurück welcher nur den Wert 1 enthält, und dem Shape Size = BATCH_SIZE
        target = torch.flatten(target)


        #Call xAI
        if local_explainable:
            limexAI.get_explanation(generated_data=fake_img, discriminator=NN_Discriminator, prediction=pred,
                            device=device, trained_data=trained_data)

        #loss = F.binary_cross_entropy(pred, target)  # loss_func(pred, target)
        criterion = nn.BCELoss()
        loss = criterion(pred, target)
        print("loss_Gen:", loss)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(NN_Generator.parameters(), 10)

        # Backprop./ Update der Gewichte des Generators
        Gen_Opt.step()

        #print("Training Gen")
        return loss.item(), fake_img

    def training(NN_Discriminator, NN_Generator, limited, train_loader, random_Tensor, num_epochs, device, lr, batchsize, weights_init, explainable, start_idx=1):

        torch.cuda.empty_cache()  # leert den Cache, wenn auf der GPU gearbeitet wird

        start_time = time.time()

        explanationSwitch = (num_epochs + 1) / 2 if num_epochs % 2 == 1 else num_epochs / 2

        NN_Generator.apply(weights_init)
        NN_Discriminator.apply(weights_init)


        NN_Discriminator.train()
        NN_Generator.train()

        if explainable:
            trained_data = Variable(next(iter(train_loader))[0])
            if device == "mps" or device== "cuda":
                trained_data = trained_data.to(device)
        else:
            trained_data = None


        # Listen für Übersicht des Fortschritts
        #R_Score = []
        #F_Score = []
        G_losses = []
        D_losses = []

        #AVG
        D_losses_avg = []

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

        local_explainable = False

        # Iteration über die Epochen
        for epoch in range(1, num_epochs+1):

            if explainable and (num_epochs-1) == explanationSwitch:
                NN_Generator.out.register_backward_hook(limexAI.explanation_hook_cifar)
            local_explainable = True

            print("Epoch:", epoch)
            # Iteration über die Bilder
            for i, data in enumerate(train_loader, 0):

                img_real, label = data
                #train_DCGAN.saves_gen_samples(img_real, epoch+start_idx, random_Tensor)

                # Trainieren des Diskrimniators
                d_loss_sum, d_loss_avg = train_DCGAN.train_discriminator(
                    NN_Discriminator, NN_Generator, img_real, Dis_Opt, random_Tensor, limited, i, batchsize, trained_data, Gen_Opt, local_explainable, device)

                # Trainieren des Generators
                g_loss, g_img = train_DCGAN.train_generator(
                    NN_Discriminator, NN_Generator, Gen_Opt, batchsize, random_Tensor, device, trained_data,local_explainable)

                # Count = i #Index/ Iterationen zählen
                #print("index:", i, "D_loss:", d_loss,"G_Loss:", g_loss)

            # Speichern des Gesamtlosses von D und G und der Real und Fake Scores
            D_losses.append(d_loss_sum)
            D_losses_avg.append(d_loss_avg)
            G_losses.append(g_loss)
            #R_Score.append(real_score)
            #F_Score.append(fake_score)

            # Ausgabe EPOCH, Loss: G und D, Scores: Real und Fake
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, loss_d_avf: {:.4f}".format(
                epoch+1, num_epochs, g_loss, d_loss_sum, d_loss_avg))

            # Speichern der generierten Samples/ Images
            train_DCGAN.saves_gen_samples(
                g_img, epoch+start_idx, random_Tensor)

        PATH_Disc= "./model/Discriminator.tar"
        PATH_Gen= "./model/Gnerator.tar"

        torch.save(NN_Discriminator.state_dict(), PATH_Disc)
        torch.save(NN_Generator.state_dict(), PATH_Gen)

        return G_losses, D_losses # R_Score, F_Score


# Train DCGAN with State-of-the-Art-Regularizationmethod

# Train LDGAN

# Radom Tensor
"""
