import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder layers
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers-1):
            self.encoder.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Linear(latent_dim, hidden_dim))
        for i in range(num_layers-1):
            self.decoder.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu, logvar = self.fc21(h), self.fc22(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        x_hat = self.fc4(h)
        return x_hat
    
    def forward(self, inputs):
        adjacency, features = inputs
        mu, logvar = self._encoder(adjacency, features)
        z = self.reparameterize(mu, logvar)
        adj_recon, ops_recon = self.decoder(z)
        return adj_recon, ops_recon, mu, logvar, z