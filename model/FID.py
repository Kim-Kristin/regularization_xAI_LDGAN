#Module - Training DCGAN

import sys

#Append needed function/module paths
sys.path.append('./model')
sys.path.append('./model/train')
sys.path.append('./model/generator')
sys.path.append('./model/discriminator')
sys.path.append('./model/inceptionnetwork')

sys.path.append('./src')
sys.path.append('./src/weightinit')
sys.path.append('./src/device')
sys.path.append('./src/dataloader')
sys.path.append('./src/param')


#from train import train_GAN
#from discriminator import DiscriminatorNetCifar10 as NN_Discriminator
from generator import GeneratorNetworkCIFAR10 as NN_Generator
import device
import dataloader
#import param
#import weightinit
import inceptionnetwork
#import torchvision.transforms as transforms
import numpy as np
from scipy.linalg import sqrtm
import torch
#import torch.nn as nn
import torchvision.utils as vutils

class CalcFID():
  # https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
  def calculate_activation_statistics(images,model,device,batch_size=128, dims=2048):
      model.eval() #Set State of the Model Evaluate (Not Training)
      act=np.empty((len(images), dims))

      batch=images.to(device)
      pred = model(batch)[0]

      # If model output is not scalar, apply global spatial average pooling.
      # This happens if you choose a dimensionality not equal 2048.
      if pred.size(2) != 1 or pred.size(3) != 1:
          pred = model.adaptive_avg_pool2d(pred, output_size=(1, 1))

      act= pred.cpu().data.numpy().reshape(pred.size(0), -1)

      mu = np.mean(act, axis=0)
      sigma = np.cov(act, rowvar=False)
      return mu, sigma

  def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2


    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
              'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))


    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

  def calculate_fretchet(images_real,images_fake,model, device):
      mu_1,std_1=CalcFID.calculate_activation_statistics(images_real,model,device)
      mu_2,std_2=CalcFID.calculate_activation_statistics(images_fake,model,device)

      """get fretched distance"""
      fid_value = CalcFID.calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
      return fid_value


  '''def FID(gan_path, Generator, testloader, device, random_Tensor):
      test_loader = testloader

      gan_point = torch.load(gan_path)
      #model = upgraded_net.simple_net_upgraded(input_channel, classes).to(device)
      G = Generator #().to(device)
      G.load_state_dict(gan_point["generator"])
      G.eval()

      # Test model
      print("Start testing...")
      fake_images =  NN_Generator(random_Tensor).detach()
      #latent_dim, n_images)

      # Load real images and resize them to the same size as the fake images
      for i, data in enumerate(test_loader,0):
        real_images, labels = data #[:batchsize]
        real_images = real_images.to(device)
        print(len(testloader), i)
        fid = CalcFID.calculate_fretchet(real_images,fake_images,inceptionnetwork.model, device)
        print(f"FID: {fid:.2f}")'''


  def trainloop(iters, epoch, num_epochs, i, netG, ngf, nz, img_list, device, dataloader, GAN_param):
    # Check how the generator is doing by saving G's output on fixed_noise
    if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        with torch.no_grad():
            fixed_noise = torch.randn(ngf, nz, 1, 1, device=device)
            fake_display = netG(fixed_noise, GAN_param).detach().cpu()
        img_list.append(vutils.make_grid(fake_display, padding=2, normalize=True))

    iters += i
