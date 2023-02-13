import torch

# Next, we'll compile the models and define the loss functions
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    #alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    batch_size, channel, height, width= real_samples.shape

    # alpha radomisiert zwischen 0 und 1 gewählt
    alpha= torch.rand(batch_size,1,1,1).repeat(1, channel, height, width).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)
    d_interpolates = D(interpolates, GAN_param=3).to(device)
    #print(d_interpolates.shape)
    fake = torch.ones(real_samples.size(0), 1).requires_grad_(False)
    fake = torch.flatten(fake).to(device)
    #print(fake.shape)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_GAN_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for GAN GP"""
    # Random weight term for interpolation between real and fake samples
    #alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    batch_size, channel, height, width= real_samples.shape

    # alpha radomisiert zwischen 0 und 1 gewählt
    alpha= torch.rand(batch_size,1,1,1).repeat(1, channel, height, width).to(device)
    #print(alpha)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)
    d_interpolates = D(interpolates, GAN_param=3).to(device)
    #print(d_interpolates.shape)
    fake = torch.ones(real_samples.size(0), 1).requires_grad_(False)
    fake = torch.flatten(fake).to(device)
    #print(fake.shape)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

