# APEGAN - test class
import sys
import numpy as np
sys.path.append('./src/param')
sys.path.append('./mode/FID')
sys.path.append('./model/train')

import dataloader
import device
import torch
from torch.autograd import Variable
from generator import GeneratorNetworkCIFAR10 as NN_Generator
import param
from FID import CalcFID
from tqdm import tqdm
import os
from torchvision.utils import save_image  # Speichern von Bildern
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def test_gan(NN_Generator, model, device, random_Tensor):
    print("Testing start")
    NN_Generator.eval()
    FID_scores_test = []
    img_list = []
    iters = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader.test_loader, 0):
            input, label = data
            input.to(device)
            # Generierung von Fake-Images
            fake_img = NN_Generator(random_Tensor, GAN_param=0).to(device)
            saves_gen_samples(
                fake_img, iters, random_Tensor,  dir_gen_samples = './outputs/fake')
            CalcFID.testloop(NN_Generator, param.g_features, param.latent_size, img_list, device, GAN_param=0)
            iters += 1
            fretchet_dist_test =  CalcFID.calculate_fretchet(input,fake_img,model, device=device) #calc FID
            FID_scores_test.append(fretchet_dist_test.item())
            break
    return FID_scores_test

def tensor_norm(img_tensors):
    # print (img_tensors)
    # print (img_tensors * NORM [1][0] + NORM [0][0])
    return img_tensors * param.NORM[1][0] + param.NORM[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Fake_Images")
    ax.imshow(make_grid(tensor_norm(images.detach()[:nmax]), nrow=8).permute(
        1, 2, 0).cpu())  # detach() : erstellt eine neue "Ansicht",
    # sodass diese Operationen nicht mehr verfolgt werden,
    # d. h. der Gradient wird nicht berechnet und der Untergraph
    # wird nicht aufgezeichnet > Speicher wird nicht verwendet

def saves_gen_samples(gen_img, idx, random_Tensor, dir_gen_samples):
    # Randomisierter Tensor wird an den Generator übergeben
    fake_img_name = "gen_img-{0:0=4d}.png".format(idx)
    os.makedirs(dir_gen_samples, exist_ok=True)
    # os.makedirs(dir_gen_samples,exist_ok=True)# Setzen von Bildbezeichnungen für die Fake_Images
    # Tensor-Normalisierung; Speichern der Fake_Images im Ordner "Outputs/dir_gen_samples/"
    save_image(tensor_norm(gen_img), os.path.join(
        dir_gen_samples, fake_img_name), nrow=8)
    show_images(gen_img)  # Plotten der Fake_Images
    print("Gespeichert")
