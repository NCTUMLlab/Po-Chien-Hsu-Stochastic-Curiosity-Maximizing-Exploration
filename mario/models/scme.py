import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class SCME(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(SCME, self).__init__()
        self.downsample = nn.AvgPool2d(2,2) #downsample to 42*42
        # Encoder
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        #self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(32*11*11,1024)
        self.fc11 = nn.Linear(1024, 288)
        self.fc12 = nn.Linear(1024, 288)

        # Decoder
        self.fc2 = nn.Linear(288, 1024)
        self.fc21 = nn.Linear(1024, 32*11*11)

        self.conv5 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        #self.conv6 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        #self.conv7 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.conv8 = nn.ConvTranspose2d(32, num_inputs, 3, stride=2, padding=1, output_padding=1)


        self.fc3 = nn.Linear(302, 256)
        self.fc41 = nn.Linear(256, 288)
        self.fc42 = nn.Linear(256, 288)
        
        self.apply(weights_init)
        self.fc1.weight.data = normalized_columns_initializer(
            self.fc1.weight.data, 0.01)
        self.fc1.bias.data.fill_(0)
        self.fc21.weight.data = normalized_columns_initializer(
            self.fc21.weight.data, 0.01)
        self.fc21.bias.data.fill_(0)
        self.fc11.weight.data = normalized_columns_initializer(
            self.fc11.weight.data, 0.01)
        self.fc11.bias.data.fill_(0)
        self.fc12.weight.data = normalized_columns_initializer(
            self.fc12.weight.data, 0.01)
        self.fc12.bias.data.fill_(0)
        self.fc2.weight.data = normalized_columns_initializer(
            self.fc2.weight.data, 0.01)
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data = normalized_columns_initializer(
            self.fc3.weight.data, 0.01)
        self.fc3.bias.data.fill_(0)
        self.fc41.weight.data = normalized_columns_initializer(
            self.fc41.weight.data, 0.01)
        self.fc41.bias.data.fill_(0)
        self.fc42.weight.data = normalized_columns_initializer(
            self.fc42.weight.data, 0.01)
        self.fc42.bias.data.fill_(0)

        self.forward_model = nn.Sequential(self.fc3, nn.ReLU(), self.fc41)

        self.train()

    def encode(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x)).view(-1, 32*11*11)
        x = F.elu(self.fc1(x))
        #x = F.elu(self.conv3(x))
        #x = F.elu(self.conv3(x)).view(-1, 32* 6* 6)
        #mu, sigma
        return self.fc11(x), self.fc12(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        if self.training:
            return mu + eps*std
        else:
            return mu
    
    def decode(self, z):
        z = F.elu(self.fc2(z))
        z = F.elu(self.fc21(z)).view(-1, 32, 11, 11)
        #z = F.elu(self.fc2(z)).view(-1, 32, 6, 6)
        z = F.elu(self.conv5(z))
        #z = F.elu(self.conv6(z))
        #z = F.elu(self.conv7(z))
        z = torch.sigmoid(self.conv8(z))
        return z
    
    def curiosity(self, x):
        x = F.elu(self.fc3(x))
        return self.fc41(x), self.fc42(x)

    def forward(self, inputs):
        st, st1, at = inputs
        xt, xt1 = self.downsample(st), self.downsample(st1)

        xt_mu, xt_logvar = self.encode(xt)
        xt1_mu, xt1_logvar = self.encode(xt1)
        zs = self.reparameterize(xt_mu, xt_logvar)
        zs1 = self.reparameterize(xt1_mu, xt1_logvar)
        xt1_hat = self.decode(zs1)

        #zs1_mu, zs1_logvar = self.curiosity(torch.cat((zs, at),1))
        #zs1_p = self.reparameterize(zs1_mu, zs1_logvar)
        '''
        kl= 0.5*torch.sum(((zs1_mu - xt1_mu)/torch.exp(0.5*zs1_logvar))**2, dim = -1) + \
            torch.sum(0.5*zs1_logvar, dim = -1) - torch.sum(0.5*xt1_logvar, dim = -1) + \
            0.5*torch.sum((torch.exp(0.5*xt1_logvar)/torch.exp(0.5*zs1_logvar))**2, dim = -1) - 0.5
        '''
        '''
        re = torch.mean(-torch.sum(xt1 * torch.log(1e-6 + xt1_hat) + (1 - xt1) * torch.log(1e-6 + 1 - xt1_hat) , dim = -1))
        KLD = torch.mean(-0.5 * torch.sum(1 + xt1_logvar - xt1_mu.pow(2) - xt1_logvar.exp(), dim = -1))
        '''
        return self.forward_model(torch.cat((zs, at),1)), zs1, xt1_hat, xt1, xt1_mu, xt1_logvar
        #return zs1_p, zs1, re, KLD
            



