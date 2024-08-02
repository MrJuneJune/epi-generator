# Number of workers for dataloader
# Doc: https://pytorch.org/docs/stable/data.html
workers = 2

# Batch size during training
# usually 64, but I didn't have much data sets which introduce more noises (more unseen datas),
# but probably less stable on changes as it is seeing more new stuff.
batch_size = 32

# Spatial size of training images. All images will be resized to this size using a transformer.
# I wanted to have bigger image, but 10 inch M1 chip is really not it lol.
image_size = 512

# Number of channels in the training images. For color images this is 3, we still normalize this tho.
nc = 3

# Size of z latent vector (i.e. size of generator input)
# Should increase this as image sizes increase sinse we need to introduce more inputs values for more sophiscated images?
nz = 600

# Size of feature maps in generator
ngf = 512

# Size of feature maps in discriminator
ndf = 512

# Number of training epochs
num_epochs = 791

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0
