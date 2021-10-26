# Import  modules
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import torch.utils.data as tordata

from tqdm.notebook import trange, tqdm

import pandas as pd
import matplotlib.pyplot as plt

# CPU or GPU
# using to(device) or cuda()
if torch.cuda.is_available():
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
  print("using cuda:", torch.cuda.get_device_name(0))
  pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class GAFDataset(tordata.Dataset):
    def __init__(self, train_amount=10000, test_amount=5000, val_amount=500, gaf_size=64, split='train',noise_rate=0.1):
        if split == 'train':
            file_name = data_path + 'dataset_train_' + str(gaf_size) + '.csv'
            dataset = pd.read_csv(file_name, header=None, nrows=train_amount)
            self.dataset = dataset
        elif split == 'test':
            file_name = data_path + 'dataset_test_' + str(gaf_size) + '.csv'
            dataset = pd.read_csv(file_name, header=None, nrows=test_amount)
            self.dataset = dataset
        elif split == 'validation':
            file_name = data_path + 'dataset_val_' + str(gaf_size) + '.csv'
            dataset = pd.read_csv(file_name, header=None, nrows=val_amount)
            self.dataset = dataset
        else: 
            noise_rate = int(noise_rate*100)
            file_name = '{}dataset_test_{}_noise_{}.csv'.format(data_path,gaf_size,noise_rate)
            dataset = pd.read_csv(file_name, header=None, nrows=test_amount)
            self.dataset = dataset
        self.split = split
        self.gaf_size = gaf_size
        print('Load {} dataset: {} GAFs, size= {}'.format(self.split, len(self.dataset), gaf_size))

    def __getitem__(self, index):
        label = int(self.dataset.iloc[index, 0])
        target = torch.zeros((5),dtype=torch.long)
        target[label] = 1
        data = torch.FloatTensor(self.dataset.iloc[index, 1:].values).view(1, self.gaf_size, self.gaf_size)
        return data, label, target  # GAF, label, onehot 

    def __len__(self):
        return len(self.dataset)

    def plot_image(self, index):
        img = self.dataset.iloc[index, 1:].values.reshape(self.gaf_size, self.gaf_size)
        plt.title("label = " + str(self.dataset.iloc[index, 0]))
        plt.imshow(img, interpolation='none', cmap='Blues')
        plt.show()
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: 
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 128,4,4
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 64,8,8
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # 32,16,16
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16,32,32
            nn.ConvTranspose2d(16, 1, 6, 2, 2, bias=False),
            nn.Tanh()
            # 1,64,64
        )
        self.apply(weights_init)
        self.linear = nn.Sequential(
            nn.Linear(100, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            nn.Linear(512, 128 * 4 * 4)
        )        
    def forward(self, noise):
        noise = noise.view(noise.size(0), 100)
        x = self.linear(noise)
        x = x.view(x.size(0), 128, 4, 4)
        x= self.model(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 1,64,64
            nn.Conv2d(1, 64, 4, 2, 2), 
            nn.LeakyReLU(0.2),
            # 64,32,32
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 128,16,16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            # 256,8,8
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 512, 4, 4
            nn.Conv2d(512, 1024, 4, 1, 0),
            nn.LeakyReLU(0.2),
        )
        self.linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 6),
        )
        self.apply(weights_init)

    def forward(self, gaf):
        x = self.model(gaf)
        x = x.view(x.size(0), 1024)
        x= self.linear(x) # batch_size, 6
        v0 = x[:, 0]
        pre_c = softmax(x[:, 1:])  
        return v0, (pre_c)
# Initialization
torch.manual_seed(manual_seed)

netG = Generator()
netG.to(device)

netD = Discriminator()
netD.to(device)

L_C  = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)

c_label = torch.LongTensor(batch_size, 5).to(device) 
  

noise = torch.FloatTensor(batch_size, 100, 1, 1).to(device)
fixed_noise = torch.FloatTensor(batch_size, 100, 1, 1).normal_(0, 1).to(device) 
fixed_noise_ = np.random.normal(0, 1, (batch_size, 100))
random_label = np.random.randint(0, 5, batch_size) 
random_onehot = np.zeros((batch_size, 5))
random_onehot[np.arange(batch_size), random_label] = 1
fixed_noise_[np.arange(batch_size), :5] = random_onehot[np.arange(batch_size)]
fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise_ = fixed_noise_.resize_(batch_size, 100, 1, 1)
fixed_noise.data.copy_(fixed_noise_) 

# Adam optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


# Training
for epoch in trange(epochs):
    for i, data in enumerate(train_loader, 0):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        netD.zero_grad()
        imgs, labels, targets = data
        real_imgs = Variable(imgs.type(Tensor), requires_grad=True)
        real_validity, real_pre_c = netD(real_imgs)
        real_grad_out = Variable(Tensor(real_imgs.size(0), 1).fill_(1.0), requires_grad=True)
        real_grad = autograd.grad(
            real_validity, real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
        loss_c_real = L_C(real_pre_c, labels)
        
        noise.data.resize_(batch_size, 100, 1, 1)
        noise.data.normal_(0, 1)
        label = np.random.randint(0, 5, batch_size) 
        noise_ = np.random.normal(0, 1, (batch_size, 100))
        label_onehot = np.zeros((batch_size, 5))
        label_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :5] = label_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, 100, 1, 1)
        noise.data.copy_(noise_)
        c_label.resize_(batch_size).copy_(torch.from_numpy(label))
        fake_imgs = netG(noise)
        fake_validity, fake_pre_c = netD(fake_imgs)
        fake_grad_out = Variable(Tensor(fake_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake_grad = autograd.grad(
            fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
        # Adversarial loss
        loss_s_D = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp
        loss_c_fake = L_C(fake_pre_c, c_label)
        loss_c_D = loss_c_real + loss_c_fake
        J_D = alpha * loss_c_D  + (1 - alpha) * loss_s_D
        J_D.backward()
        optimizerD.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        netG.zero_grad()
        fake_imgs = netG(noise)
        fake_validity, fake_pre_c = netD(fake_imgs)
        loss_s_G = -torch.mean(fake_validity)
        loss_c_G = L_C(fake_pre_c, c_label)
        J_G = alpha * loss_c_G - (1 - alpha) * loss_s_G
        J_G.backward()
        optimizerG.step()
#   do checkpointing
    torch.save(netG.state_dict(), '%snetG_epoch_%d.pth' % (model_path, epoch + 1))
    torch.save(netD.state_dict(), '%snetD_epoch_%d.pth' % (model_path, epoch + 1))