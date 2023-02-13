# References:
# 1. https://medium.datadriveninvestor.com/xai-with-lime-for-cnn-models-5560a486578
# 2. https://github.com/explainable-gan/XAIGAN

# import packages


import numpy as np
from copy import deepcopy
import torch
from torch.nn import functional as F
from lime import lime_image
from torch import Tensor, from_numpy, randn, full
import torch.nn as nn
from torch.autograd.variable import Variable

# defining global variables
global values
global discriminatorLime


def get_explanation(generated_data, discriminator, prediction, device,trained_data=None):
    '''
    This function calculates the explanation for given generated images using the desired xAI systems and the
    :param generated_data: data created by the generator
    :type generated_data: torch.Tensor
    :param discriminator: the discriminator model
    :type discriminator: torch.nn.Module
    :param prediction: tensor of predictions by the discriminator on the generated data
    :type prediction: torch.Tensor
    :param XAItype: the type of xAI system to use. One of ("shap", "lime", "saliency")
    :type XAItype: str
    :param cuda: whether to use gpu
    :type cuda: bool
    :param trained_data: a batch from the dataset
    :type trained_data: torch.Tensor
    :param data_type: the type of the dataset used. One of ("cifar", "mnist", "fmnist")
    :type data_type: str
    :return:
    :rtype:
    '''

    # initialize temp values to all 1s
    temp = values_target(size=generated_data.size(), value=1.0, device=device)

    # mask values with low prediction
    mask = (prediction < 0.5).view(-1)
    indices = (mask.nonzero(as_tuple=False)).detach().cpu().numpy().flatten().tolist()



    data = generated_data[mask, :]
    data.to(device)
    #print ("trained data size ", trained_data.size())
    #print ("data size ", data.unsqueeze(0).size())
    #print ("indices size ", len(indices))
    #exit(0)


    if len(indices) > 1:

        print("Explanation with LIME")

        explainer = lime_image.LimeImageExplainer()
        global discriminatorLime
        discriminatorLime = deepcopy(discriminator)
        discriminatorLime.cpu()
        discriminatorLime.eval()
        for i in range(len(indices)):
            tmp = data[i, :].detach().cpu().numpy()
            #print(tmp.shape)
            tmp = np.reshape(tmp, (64, 64, 3)).astype(np.double)
            #tmp =
            exp = explainer.explain_instance(tmp, batch_predict_cifar, num_samples=100)
            temp[indices[i], :] = mask.clone().detach()
        del discriminatorLime

    temp = temp.to(device)
    set_values(normalize_vector(temp))

def explanation_hook_cifar(module, grad_input, grad_output):
    """
    This function creates explanation hook which is run every time the backward function of the gradients is called
    :param module: the name of the layer
    :param grad_input: the gradients from the input layer
    :param grad_output: the gradients from the output layer
    :return:
    """
    # get stored mask
    temp = get_values()

    # multiply with mask
    new_grad = grad_input[0] + 0.2 * (grad_input[0] * temp)

    return (new_grad, )


def normalize_vector(vector: torch.tensor):
    #normalize np array to the range of [0,1] and returns as float32 values'''
    vector = vector - vector.min()
    vector = vector/ vector.max()
    vector[torch.isnan(vector)] = 0
    return vector.type(torch.float32)


def get_values():
    """ get global values """
    global values
    return values


def set_values(x: np.array):
    """ set global values """
    global values
    values = x

def batch_predict_cifar(images):
    """ function to use in lime xAI system for CIFAR10"""
    # stack up all images
    images = np.transpose(images, (0, 3, 1, 2))
    batch = torch.stack([i for i in torch.Tensor(images)], dim=0)
    logits = discriminatorLime(batch, GAN_param=1)
    probs = F.softmax(logits, dim=0).view(-1).unsqueeze(1)
    return probs.detach().numpy()



def values_target(size: tuple, value: float, device):
    #returns tensor filled with value of given size
    result = Variable(full(size=size, fill_value=value))
    result = result.to(device)
    return result

