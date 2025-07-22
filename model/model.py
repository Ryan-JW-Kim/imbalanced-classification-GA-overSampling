import torch.nn as nn
import torch

class ConditionalVAE(nn.Module):
	def __init__(self, input_dim, h1, h2, latent_dim):
		super(ConditionalVAE, self).__init__()
		# hidden_dim = input_dim + input_dim // 2  # Overcomplete expansion
		
		# Encoder
		self.enc_fc1 = nn.Linear(input_dim, h1)
		self.enc_fc2 = nn.Linear(h1, h2)
		self.fc_mu = nn.Linear(h2, latent_dim)
		self.fc_logvar = nn.Linear(h2, latent_dim)

		# Decoder
		self.dec_fc1 = nn.Linear(latent_dim + 1, h2)
		self.dec_fc2 = nn.Linear(h2, h1)
		self.dec_fc3 = nn.Linear(h1, input_dim)

	def encode(self, x):
		h = torch.relu(self.enc_fc1(x))
		h = torch.relu(self.enc_fc2(h))
		return self.fc_mu(h), self.fc_logvar(h)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def decode(self, z, y):
		y = y.unsqueeze(1)
		z = torch.cat([z, y], dim=1)
		h = torch.relu(self.dec_fc1(z))
		h = torch.relu(self.dec_fc2(h))
		return self.dec_fc3(h)

	def forward(self, x, label):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z, label), mu, logvar
