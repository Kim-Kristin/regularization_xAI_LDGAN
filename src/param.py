import torch
import random
import device

# Root directory for dataset
dataroot = "data/"
# choice between gradient scaling=1 and gradient clipping=2
choice_gc_gs = 2
# Number Iteration for Critic (Discriminator)
N_CRTIC = 5
#Clipping weight
WEIGHT_CLIP = 0.01
# Limitierung des Discriminators =True
limited=True
# Number of workers for dataloader
num_workers = 0
# Batch size during training
batch_size = 64
# Spatial size of training images. All images will be resized to this
# size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
in_channel = 3
# Size of z latent vector (i.e. size of generator input)
latent_size = 100
# Size of feature maps in generator
g_features = 64
# Size of feature maps in discriminator
d_features = 64
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Normalisierung mit 0.5 Mittelwert und Standardabweichung für alle drei
NORM = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
# Set random seed for reproducibility
randomseed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", randomseed)
random.seed(randomseed)
torch.manual_seed(randomseed)
#Random Tensor
random_Tensor = torch.randn(batch_size, latent_size, 1, 1, device= device.device)

# Split
# Train, Validation, Test
TEST_SPLIT = 0.2
VALID_SPLIT = 0.7
TRAIN_SPLIT = (1-(TEST_SPLIT+VALID_SPLIT))

SPLIT_AUFTEILUNG = {TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT}
