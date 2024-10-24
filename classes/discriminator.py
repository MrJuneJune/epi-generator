import torch.nn as nn
from constants import ndf, nc


# going to assume input image is 2048 x 1024 rather than 1980 x 1080 for numerical simplicty
# 13 109 697 parameters (ndf = 64) -- I think it should be smaller 
# Conv: ~ 4.7M params , MLP: ~ 8.4M params. Scales quadratically with ndf.
# Probably want dropout at some point to stop overfitting
class Discriminator(nn.Module):
    def __init__(self, ngpu, dropout = 0.0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        #Convolutional part
        self.conv = nn.Sequential(
            # input is ``(nc) x 2048 x 1024``
            DoubleConv(nc, ndf, dropout=dropout),
            nn.MaxPool2d(4),
            # state size. ``(ndf) x 512 x 256``
            DoubleConv(ndf, ndf*2, dropout=dropout),
            nn.MaxPool2d(4),
            # state size. ``(ndf*2) x 128 x 64``
            DoubleConv(ndf*2, ndf*4, dropout=dropout),
            nn.MaxPool2d(4),
            # state size. ``(ndf*4) x 32 x 16``
            DoubleConv(ndf*4, ndf*8, dropout=dropout),
            nn.MaxPool2d(4),
            # state size. ``(ndf*8) x 8 x 4``
        )

        # Dense layers to process the CNN output
        self.MLP = nn.Sequential(
            nn.LayerNorm(ndf * 8 * 8 * 4),
            nn.Linear(ndf * 8 * 8 * 4, ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf * 8, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, input):
        x = self.conv(input)
        x = x.flatten()
        x = self.MLP(x)
        x = nn.functional.sigmoid(x)
        return x
    
# 2 convolutional layers with gelu nonlinearity in between, 
# first layer increases channel number (unless otherwise specified)
# best practices (I think) to use group norm rather than batch norm 
# https://medium.com/@zljdanceholic/groupnorm-then-batchnorm-instancenorm-layernorm-e2b2a1d350a0
class DoubleConv(nn.Module):

    def __init__(self, c_in, c_out, c_mid = None, dropout = 0.0):
        super().__init__()
        if not c_mid:
            c_mid = c_out

        self.main = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding='same', bias=False),
            nn.GroupNorm(1,c_mid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_mid, c_out, kernel_size=3, padding='same', bias=False),
            nn.GroupNorm(1,c_out),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        return self.main(input)




