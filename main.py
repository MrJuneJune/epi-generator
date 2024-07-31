import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as tvutils
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import matplotlib.pyplot as plt

# Constants
from constants import *
from classes import *
from datasets import init_dataloader
from helpers import weights_init

# Use GPU if it can
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
dataloader = init_dataloader()

def create_nn_optimizers(device):
    # Create the generator
    netG = Generator(ngpu).to(device)
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)
    
    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)
    
    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)
    
    # Handle multi-GPU if desired, but I don't have because I am poor.
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Load weights if available
    # Saving models incase my computer blows up.
    model_save_path = "./models"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    generator_weights = os.path.join(model_save_path, f'generator_new.pth')
    discriminator_weights = os.path.join(model_save_path, f'discriminator_new.pth')
    
    if os.path.exists(generator_weights):
        netG.load_state_dict(torch.load(generator_weights, weights_only=True))
        print("Loaded generator weights")
    
    if os.path.exists(discriminator_weights):
        netD.load_state_dict(torch.load(discriminator_weights, weights_only=True))
        print("Loaded discriminator weights")

    return netD, netG, optimizerD, optimizerG, generator_weights, discriminator_weights

def train(dataloader):
    print("Starting Training Loop...")
    iters = 0
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
    
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
    
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
    
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
    
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
    
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                # Save into images to see how it improves over time.
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                # Save model weights periodically
                print(f"iters: {iters}\tSaving models..")
                torch.save(netG.state_dict(), generator_weights)
                torch.save(netD.state_dict(), discriminator_weights)

            iters += 1

def plot_learning_graph():
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def show_generated_images(generator, num_images=16):
    noise = torch.randn(num_images, nz, 1, 1, device=device)
    fake_images = generator(noise).detach().cpu()
    grid = tvutils.make_grid(fake_images, padding=2, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()

def show_real_images():
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


netD, netG, optimizerD, optimizerG, generator_weights, discriminator_weights = create_nn_optimizers(device)

# Initialize the ``BCELoss`` function
# Optimizer for discriminator.
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(ngf, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
# we can actually make this into 0.9 and 0.1 if we want to try smoothing the curve a bit, but probably better to change the lr.
real_label = 1.
fake_label = 0.

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []

if __name__=="__main__":
    train(dataloader)
    plot_learning_graph()
    show_real_images()
    show_generated_images(netG)
