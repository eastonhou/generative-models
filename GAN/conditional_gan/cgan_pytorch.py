import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from torchvision import datasets, transforms

mb_size = 64
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(mnist, batch_size=mb_size, shuffle=True)
dataiter = iter(dataloader)
Z_dim = 100
X_dim = mnist.data.shape[2]*mnist.data.shape[1]
y_dim = 16
h_dim = 128
cnt = 0
lr = 1e-3

""" ==================== GENERATOR ======================== """
class Generator(torch.nn.Module):
    def __init__(self):
        super(__class__, self).__init__()
        self.embeddings = torch.nn.Embedding(10, y_dim)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(Z_dim+y_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, X_dim),
            torch.nn.Sigmoid())

    def forward(self, z, c):
        c = self.embeddings(c)
        inputs = torch.cat([z, c], 1)
        y = self.layers(inputs)
        return y

""" ==================== DISCRIMINATOR ======================== """
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(__class__, self).__init__()
        self.embeddings = torch.nn.Embedding(10, y_dim)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(X_dim+y_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, 1),
            torch.nn.Sigmoid())

    def forward(self, X, c):
        c = self.embeddings(c)
        inputs = torch.cat([X, c], 1)
        y = self.layers(inputs)
        return y


generator = Generator()
discriminator = Discriminator()
G_params = generator.parameters()
D_params = discriminator.parameters()
G_solver = optim.Adam(G_params, lr=1e-3)
D_solver = optim.Adam(D_params, lr=1e-3)
""" ===================== TRAINING ======================== """

def reset_grad():
    G_solver.zero_grad()
    D_solver.zero_grad()

ones_label = torch.ones(mb_size, 1)
zeros_label = torch.zeros(mb_size, 1)

def sample_X():
    global dataiter
    try:
        X, Y = dataiter.next()
    except:
        dataiter = iter(dataloader)
        X, Y = dataiter.next()
    return X, Y

for it in range(100000):
    # Sample data
    X, c = sample_X()
    z = torch.randn(X.shape[0], Z_dim)
    X = X.view(X.shape[0], -1)

    # Dicriminator forward-loss-backward-update
    G_sample = generator(z, c)
    D_real = discriminator(X, c)
    D_fake = discriminator(G_sample, c)

    #D_loss_real = nn.binary_cross_entropy(D_real, ones_label[:X.shape[0]])
    #D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label[:X.shape[0]])
    #D_loss = D_loss_real + D_loss_fake
    D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
    D_loss.backward()
    D_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    # Generator forward-loss-backward-update
    z = torch.randn(X.shape[0], Z_dim)
    G_sample = generator(z, c)
    D_fake = discriminator(G_sample, c)

    #G_loss = nn.binary_cross_entropy(D_fake, ones_label[:X.shape[0]])
    G_loss = -torch.mean(torch.log(D_fake))

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'.format(it, D_loss.data.numpy(), G_loss.data.numpy()))

        c = torch.arange(0, mb_size) % 10
        samples = generator(z, c).detach().numpy()[:16]

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

        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
