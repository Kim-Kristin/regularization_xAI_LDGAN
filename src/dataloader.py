import os
import enum
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import param
from torchvision import datasets
import torch.utils.data as DataLoader
from torchvision.utils import make_grid
import torch.nn as nn  # Neuronales Netz
import torch.optim as optim  # Optimierungs-Algorithmen
from torchvision.utils import save_image  # Speichern von Bildern
import torch.nn.functional as F  # Loss
import device
from device import DeviceDataLoader

# Reference: https://github.com/Dianevera/heart-prediction/blob/94f2b9919d78a47ef3ee4d711e879ab1d08b273c/heartpredictions/LSTM/create_dataloaders.py

# StandardPackage

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms

# CustomPackages
import param


def dataloader(dataset, BATCH_SIZE, split_aufteilung, display_informations=True, num_of_worker=param.num_workers, random_seed=param.randomseed):
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)
    # calculate lengths per dataset without consideration Split_Aufteilung
    lengths = [round(len(dataset) * split) for split in split_aufteilung]

    r = 0

    for i in range(len(lengths)):
        #print(len(lengths))
        r_tmp = lengths[i] % 3  # Value an der Stelle i modulo 3
        lengths[i] = lengths[i] - r_tmp
        r += r_tmp
        # print(r)
    lengths[2] += r

    # Calculation of the dataset-sizes
    train = torch.utils.data.Subset(dataset, range(0, lengths[0]))
    validation = torch.utils.data.Subset(
        dataset, range(lengths[0], lengths[0] + lengths[1]))
    test = torch.utils.data.Subset(dataset, range(
        lengths[0] + lengths[1], lengths[0] + lengths[1] + lengths[2]))

    # train loader
    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        worker_init_fn=random_seed,
        num_workers=0,  # num_Worker = 0 because MemoryError
        persistent_workers=False,
        pin_memory=True
    )

    # validation loader
    validation_dataloader = torch.utils.data.DataLoader(
        validation,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        worker_init_fn=random_seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True
    )

    # test loader
    test_dataloader = torch.utils.data.DataLoader(
        test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        worker_init_fn=random_seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True
    )

    # show length of the train, validation and testloader
    if display_informations:
        print(f'Total dataset: {len(train_dataloader) + len(validation_dataloader) + len(test_dataloader)}, '
              f'train dataset: {len(train_dataloader)}, val dataset: {len(validation_dataloader)}, test_dataset: {len(test_dataloader)}')

    return train_dataloader, validation_dataloader, test_dataloader


# Transformer
transform = transforms.Compose([
    # Resize der Images auf 64 der kürzesten Seite; Andere Seite wird
    transforms.Resize(param.image_size),
    # skaliert, um das Seitenverhältnis des Bildes beizubehalten.
    # Zuschneiden auf die Mitte des Images, sodass ein quadratisches Bild mit 64 x 64 Pixeln entsteht
    transforms.CenterCrop(param.image_size),
    # Umwandeln in einen Tensor (Bildern in numerische Werte umwandeln)
    transforms.ToTensor(),
    transforms.Normalize(*param.NORM)])          # Normalisierung Mean & Standardabweichung von 0.5 für alle Channels
# (Anzahl: 3 für farbige Bilder)
# Pixelwerte liegen damit zwischen (-1;1)

# CIFAR-10
train_dataset_CIFAR10 = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                                                     transform=transform)

test_dataset_CIFAR10 = datasets.CIFAR10(root='./data', train=False, download=True,
                                        transform=transform)
# Concat Train and Test Dataset --> Whole Data
'''CIFAR-10'''
dataset_CIFAR10 = ConcatDataset([train_dataset_CIFAR10, test_dataset_CIFAR10])


train_dataloader_CIFAR10, validation_dataloader_CIFAR10, test_dataloader_CIFAR10 = dataloader(
    dataset_CIFAR10, param.batch_size, param.SPLIT_AUFTEILUNG, param.num_workers)

"""
trainset = torchvision.datasets.CIFAR10(root=param.dataroot, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=param.batch_size,
                                          shuffle=True, num_workers=0)
train_loader = trainloader.to(device.device)

testset = torchvision.datasets.CIFAR10(root=param.dataroot, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=param.batch_size,
                                         shuffle=False, num_workers=0)

# classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')"""

#classes = dataset_CIFAR10.classes
# print(classes)

'''
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(param.batch_size)))
'''

# Dataloader auf dem verfügbaren Device
train_loader = device.DeviceDataLoader(train_dataloader_CIFAR10, device.device)
test_loader = device.DeviceDataLoader(test_dataloader_CIFAR10, device.device)

# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""for i, data in enumerate(train_loader):
    real_img, label = data

    fake_img_name = "gen_img-{0:0=4d}.png".format(i)
    dir_gen_samples = './outputs/dir_gen_samples'
    os.makedirs('./outputs/dir_gen_samples', exist_ok=True)
    # os.makedirs(dir_gen_samples,exist_ok=True)# Setzen von Bildbezeichnungen für die Fake_Images
    # Tensor-Normalisierung; Speichern der Fake_Images im Ordner "Outputs/dir_gen_samples/"
    save_image(real_img, os.path.join(
        dir_gen_samples, fake_img_name), nrow=8)
    print("Gespeichert")
    break"""
