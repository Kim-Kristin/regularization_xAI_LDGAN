import numpy as np
import torch
from torch.autograd import Variable
#from torchvision.models import inception_v3
import generator
import discriminator
from tqdm import tqdm

# First, we'll define a function for generating a batch of fake images
def generate_fake_images(generator,random_Tensor, device): # latent_dim, n_images, device):
  # Sample random points in the latent space
  random_latent_vectors = random_Tensor #torch.randn(n_images, latent_dim).to(device)

  # Decode them to fake images
  fake_images = generator(random_latent_vectors)

  return fake_images.detach().cpu().numpy()

# Next, we'll define a function for calculating the FID
def calculate_fid(real_images, fake_images, G, device):
  # Calculate the Inception Score of the real and fake images
  real_inception_score = get_inception_score(real_images, G, device)
  fake_inception_score = get_inception_score(fake_images, G, device)

  # Calculate the FID
  fid = (real_inception_score[1] + np.square(real_inception_score[0] - fake_inception_score[0]))

  return fid

# We'll also define a function for calculating the Inception Score
def get_inception_score(images, G, device):
      # Load the Inception model
  inception_model = G.to(device)
  inception_model.eval()

  # Transform the images to tensors and feed them through the model
  images = (images * 255).astype(np.uint8)
  images = torch.from_numpy(images).to(device)
  images = Variable(images, requires_grad=False)
  with torch.no_grad():
    logits = inception_model(images)
  probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

  # Calculate the mean and standard deviation of the predicted class probabilities
  mean = np.mean(probs, axis=0)
  std = np.std(probs, axis=0)

  return mean, std


def test(gan_path, Generator, testloader, device, input_channel, random_Tensor, batchsize): #latent_dim, n_images=10000):
    test_loader = testloader

    gan_point = torch.load(gan_path)
    #model = upgraded_net.simple_net_upgraded(input_channel, classes).to(device)
    G = Generator(input_channel).to(device)
    G.load_state_dict(gan_point["generator"])
    G.eval()

    # Test model
    print("Start testing...")
    fake_images = generate_fake_images(G, random_Tensor, device) #latent_dim, n_images)

    real_images = np.array([np.array(image.resize((batchsize, batchsize))) for image in real_images])
    # Calculate the FID
    fid = calculate_fid(real_images, fake_images, G, device)
    print(f"FID: {fid:.2f}")


    """for input, label in tqdm(test_loader, total=len(test_loader), leave=False):
        input, label = Variable(input.to(device)), Variable(label.to(device))

        normal_acc = attack_and_eval.evaluation(fmodel, input, label)

        input_adv, _, success = attack(fmodel, input , label, epsilons=eps)

        adv_acc = attack_and_eval.evaluation(fmodel, input_adv, label)

        input_ape = G(input_adv)

        ape_acc = attack_and_eval.evaluation(fmodel, input_ape, label)
        n += label.size(0)
    print("Accuracy: normal {:.6f}".format(
        normal_acc / n * 100))
    print("Accuracy: " + str(attack_name) + " {:.6f}".format(
        adv_acc / n * 100))
    print("Accuracy: APEGAN {:.6f}".format(
        ape_acc / n * 100))x<
"""
