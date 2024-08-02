import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim

# Constants
from constants import nz, ngf, num_epochs, lr, beta1, ngpu
from classes import Generator, Discriminator
from datasets import init_dataloader
from helpers import weights_init

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dataloader
dataloader = init_dataloader()

def initialize_models_and_optimizers(device):
    """Initialize models, optimizers, and load weights if available."""
    # Create the generator and discriminator
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)

    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Handle multi-GPU if available
    if device.type == 'cuda' and ngpu > 1:
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Model save path
    model_save_path = "./models"
    os.makedirs(model_save_path, exist_ok=True)

    generator_weights = os.path.join(model_save_path, 'generator_new_512.pth')
    discriminator_weights = os.path.join(model_save_path, 'discriminator_new_512.pth')

    # Load weights if available
    if os.path.exists(generator_weights):
        netG.load_state_dict(torch.load(generator_weights, weights_only=True))
        print("Loaded generator weights")
    
    if os.path.exists(discriminator_weights):
        netD.load_state_dict(torch.load(discriminator_weights, weights_only=True))
        print("Loaded discriminator weights")

    return netD, netG, optimizerD, optimizerG, generator_weights, discriminator_weights

def train(dataloader, netD, netG, optimizerD, optimizerG, generator_weights, discriminator_weights):
    """Train the GAN."""
    print("Starting Training Loop...")
    criterion = nn.BCELoss()
    real_label = 1.0
    fake_label = 0.0
    fixed_noise = torch.randn(ngf, nz, 1, 1, device=device)
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Update Discriminator with real and fake data
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update Generator
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                print(f"iters: {iters}\tSaving models..")
                torch.save(netG.state_dict(), generator_weights)
                torch.save(netD.state_dict(), discriminator_weights)

            iters += 1

    return img_list, G_losses, D_losses

def plot_losses(G_losses, D_losses):
    """Plot the training losses for Generator and Discriminator."""
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def show_generated_images(generator, num_images=16):
    """Display generated images."""
    noise = torch.randn(num_images, nz, 1, 1, device=device)
    fake_images = generator(noise).detach().cpu()
    grid = vutils.make_grid(fake_images, padding=2, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()

def show_real_images(dataloader):
    """Display real images from the training dataset."""
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    netD, netG, optimizerD, optimizerG, generator_weights, discriminator_weights = initialize_models_and_optimizers(device)
    img_list, G_losses, D_losses = train(dataloader, netD, netG, optimizerD, optimizerG, generator_weights, discriminator_weights)
    plot_losses(G_losses, D_losses)
    show_real_images(dataloader)
    show_generated_images(netG)

