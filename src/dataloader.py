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


trainset = torchvision.datasets.CIFAR10(root=param.dataroot, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=param.batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root=param.dataroot, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=param.batch_size,
                                         shuffle=False, num_workers=0)

# classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

classes = trainset.classes
print(classes)

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
train_loader = device.DeviceDataLoader(trainloader, device.device)

# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

train_batch = next(iter(train_loader))  # Iterieren über Dataloader
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Original_Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(train_batch[0].to(device.device)[
           :64], padding=2, normalize=True).cpu(), (1, 2, 0)))  # (1,2,0): Ausrichtung der Bilder
