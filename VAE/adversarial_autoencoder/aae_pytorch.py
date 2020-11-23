import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from torchvision import datasets, transforms
#from tensorflow.examples.tutorials.mnist import input_data

transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(mnist, batch_size=100, shuffle=True)
dataiter = iter(dataloader)
#mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 100
z_dim = 5
X_dim = mnist.data.shape[2]*mnist.data.shape[1]
#y_dim = mnist.data.shape[1]
h_dim = 128
cnt = 0
lr = 1e-3


# Encoder
encoder = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, z_dim)
).to(0)

# Decoder
decoder = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
).to(0)

# Discriminator
descriminator = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
    torch.nn.Sigmoid()
).to(0)


def reset_grad():
    encoder.zero_grad()
    decoder.zero_grad()
    descriminator.zero_grad()


def sample_X(size, include_y=False):
    global dataiter
    try:
        X, Y = dataiter.next()
    except:
        dataiter = iter(dataloader)
        X, Y = dataiter.next()
    if include_y: return X, Y
    else: return X


encoder_optim = optim.Adam(encoder.parameters(), lr=lr)
decoder_optim = optim.Adam(decoder.parameters(), lr=lr)
descriminator_optim = optim.Adam(descriminator.parameters(), lr=lr)


for it in range(1000000):
    X = sample_X(mb_size).to(0)
    X = X.view(X.shape[0], -1)

    """ Reconstruction phase """
    z_sample = encoder(X)
    X_sample = decoder(z_sample)

    recon_loss = nn.binary_cross_entropy(X_sample, X)

    recon_loss.backward()
    decoder_optim.step()
    encoder_optim.step()
    reset_grad()

    """ Regularization phase """
    # Discriminator
    z_real = torch.randn(mb_size, z_dim).to(0)
    z_fake = encoder(X)

    D_real = descriminator(z_real)
    D_fake = descriminator(z_fake)

    D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))

    D_loss.backward()
    descriminator_optim.step()
    reset_grad()

    # Generator
    z_fake = encoder(X)
    D_fake = descriminator(z_fake)

    G_loss = -torch.mean(torch.log(D_fake))

    G_loss.backward()
    encoder_optim.step()
    reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        print(f'Iter-{it}; D_loss: {D_loss.item():.4}; G_loss: {G_loss.item():.4}; recon_loss: {recon_loss.item():.4}')

        samples = decoder(z_real).detach().cpu().numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'
                    .format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
