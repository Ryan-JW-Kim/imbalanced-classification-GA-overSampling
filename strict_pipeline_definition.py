from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import ranksums

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import Sampling
from pymoo.operators.crossover.hux import HUX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from collections import defaultdict

from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os

import wandb

class TabularDataset(Dataset):
	def __init__(self, x_synthetic, x_true):
		self.x = x_synthetic
		self.y = x_true
	def __len__(self):
		return self.x.shape[0]
	def __getitem__(self, ind):
		x = self.x[ind]
		y = self.y[ind]
		return x, y

class ConditionalVAE(nn.Module):
	def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
		super(ConditionalVAE, self).__init__()
		self.fc1 = nn.Linear(input_dim + label_dim, hidden_dim)
		self.fc_mu = nn.Linear(hidden_dim, latent_dim)
		self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
		self.fc2 = nn.Linear(latent_dim + label_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, input_dim)

	def encode(self, x, y):
		# Concatenate input and label
		x = torch.cat([x, y], dim=1)
		h = torch.relu(self.fc1(x))
		return self.fc_mu(h), self.fc_logvar(h)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def decode(self, z, y):
		# Concatenate latent vector and label
		z = torch.cat([z, y], dim=1)
		h = torch.relu(self.fc2(z))
		return self.fc3(h)

	def forward(self, x, y):
		mu, logvar = self.encode(x, y)
		z = self.reparameterize(mu, logvar)
		return self.decode(z, y), mu, logvar
	
class cVAE_GA:
	def __init__(self, x_train, y_train, x_validation, y_validation, hyperparams):
		
		minority_label = pd.DataFrame(y_train).value_counts().argmin()
		minority_indices = np.where(y_train==minority_label)[0]
		minority_features = x_train[minority_indices]
		minority_labels = y_train[minority_indices]
		
		self.hyperparams = hyperparams

		self.input_dim = x_train[0].shape[0]
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def train_cVAE(self, x, y, lr, epochs, batch_size, beta, logger=None, cvae=None):

		if cvae is None:
			cvae = ConditionalVAE(self.input_dim, 1, self.input_dim//2, 2).to(self.device)

		train_set = TabularDataset(torch.from_numpy(x), torch.from_numpy(y))
		train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
		optimizer = optim.Adam(cvae.parameters(), lr=lr)
		
		for epoch in range(epochs):
			cvae.train()
			total_loss = 0
			for batch in train_loader:
				x_batch = batch[0].to(self.device).float()
				y_batch = batch[1].to(self.device).float().unsqueeze(1)
				
				recon, mu, logvar = cvae(x_batch, y_batch)
				recon_loss = nn.MSELoss()(recon, x_batch)
				kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
				loss = recon_loss + (kl_div*beta)
		
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				total_loss += loss.item()
			if logger is not None:
				logger.log({"training loss": total_loss / len(train_loader)})
			
			return cvae
	
	def create_classifier(self, x1, y1, x2, y2):
		metrics = defaultdict(lambda: -1)

		if x1[0].shape[0] > self.hyperparams['N']:
			model = KNeighborsClassifier(n_neighbors=self.hyperparams['N'])
			model.fit(x1, y1)
			y_pred = model.predict(x2)
			metrics['AUC'] = roc_auc_score(y2, y_pred)
		
		return metrics

	def generate_samples(self, x_samples, y_samples, variance, cvae):
		cvae.eval()
		