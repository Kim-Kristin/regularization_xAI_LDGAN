import os
from torchvision.utils import save_image  # Speichern von Bildern
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def plot_metrics(reg_model,G_losses, D_losses):
    path = "./outputs/metrics/"
    os.makedirs(path, exist_ok=True)
    EPOCH_COUNT_G= range(1,len(G_losses)+1) # Anzahl der Epochen vom Gen.
    EPOCH_COUNT_D= range(1,len(D_losses)+1) # Anzahl der Epochen vom Dis.
    fig = plt.figure(figsize=(10,5))
    plt.title("LOSS: Generator und Discriminator während dem Training")
    plt.plot(EPOCH_COUNT_G, G_losses,"r-", label="Gen")
    plt.plot(EPOCH_COUNT_D,D_losses,"b-", label="Dis")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.legend()
    name = "loss"+reg_model+".png"
    fig.savefig(path+name, dpi=fig.dpi)

def plot_FID(reg_model,FID):
    path = "./outputs/metrics/"
    os.makedirs(path, exist_ok=True)
    FID_scores= range(1,len(FID)+1) # Anzahl der Epochen vom Gen.
    fig = plt.figure(figsize=(10,5))
    plt.title("FID während dem Training")
    plt.plot(FID_scores, FID,"r-", label="FID")
    plt.xlabel("EPOCH")
    plt.ylabel("Score")
    plt.legend()
    name = "fid"+reg_model+".png"
    fig.savefig(path+name, dpi=fig.dpi)
