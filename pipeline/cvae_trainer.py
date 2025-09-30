from model.model import ConditionalVAE
from model.dataset import TabularDataset
from model.utils.visualization import PCA_plot, PCA_plot_rare_on_top
from model.utils.optimization import *
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from matplotlib.cm import get_cmap
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, balanced_accuracy_score

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from itertools import product
from imblearn.over_sampling import (
	SMOTE,
	ADASYN,
	BorderlineSMOTE,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mask_features(x, min_mask: int = 1, max_mask: int = 3):
	x_masked = x.clone()
	for i in range(x.size(0)):
		k = torch.randint(min_mask, max_mask + 1, (1,)).item()
		idx = torch.randperm(x.size(1))[:k]
		x_masked[i, idx] = 0
	return x_masked

def train_cvae(x_train, y_train):
	num_features = x_train[0].shape[0]

	###############################################
	# Calculate the 5 nearest neighbours of each sample.
	# If the neighbors share a label, use either sample
	# interchangably during training to calculate loss.
	###############################################
	nearest_neighbours = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(x_train)
	dist, idx = nearest_neighbours.kneighbors(x_train)
	dist = dist[:, 1:]
	idx  = idx[:, 1:]
	knn_features = [x_train[row_idx] for row_idx in idx]
	knn_labels = [y_train[row_idx] for row_idx in idx]

	input_set, recon_set, labels = [], [], []
	for s_idx, sample in enumerate(x_train):
		for n_idx, neighbouring_sample in enumerate(knn_features[s_idx]):
			if y_train[s_idx] == knn_labels[s_idx][n_idx]:
				input_set.append(sample)
				recon_set.append(neighbouring_sample)
				labels.append(y_train[s_idx])

	################################################
	# Define the cVAE model to be an over complete
	# autoencoder with 20 latent dimensions.
	################################################
	h1 = num_features + (num_features//2)
	h2 = num_features * 2
	latent_dim = 20
	cvae = ConditionalVAE(input_dim=num_features, h1=h1, h2 = h2, latent_dim=latent_dim).to(device)

	################################################
	# Pretraining the cVAE, each sample is used to 
	# calculate the reconstruction loss.
	################################################
	epochs = 900
	batch_size = 32
	lr = 1e-3
	beta = 0.8
	data = TabularDataset(x_train, x_train, y_train)
	loader = DataLoader(data, batch_size=batch_size, shuffle=True)

	cvae.train()
	opt = optim.Adam(cvae.parameters(), lr=lr)
	for epoch in range(1, epochs + 1):
		running = 0
		for encode_in, decode_comp, label in loader:
			# In this training loop xb == yb
			xb = encode_in.float().to(device)
			yb = decode_comp.float().to(device)
			# label is the class corresponding to xb and yb
			label = label.float().to(device)
			recon, mu, logvar = cvae(xb, label)
			recon_loss = nn.MSELoss(reduction='sum')(recon, yb)
			kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
			loss = (recon_loss + beta * kl_div) / xb.size(0)   # per-batch average

			opt.zero_grad()
			loss.backward()
			opt.step()

			running += loss.item()

	################################################
	# Extend the training to now use the nearest 
	# neighbors to calculate reconstruction loss aswell
	################################################

	epochs = 400
	data = TabularDataset(
		np.array(input_set), 
		np.array(recon_set), 
		np.array(labels)
	)
	loader = DataLoader(data, batch_size=batch_size, shuffle=True)

	cvae.train()
	opt = optim.Adam(cvae.parameters(), lr=lr)
	for epoch in range(1, epochs + 1):
		for encode_in, decode_comp, label in loader:
			xb = encode_in.float().to(device)
			# Apply random masking to the encoder input
			xb_masked = mask_features(xb)
			yb = decode_comp.float().to(device)
			label = label.float().to(device)
			# Calculate the reconstruction of yb (which may be either xb or xb's neighbor)
			# and the label corresponding to xb and yb
			recon, mu, logvar = cvae(xb_masked, label)
			recon_loss = nn.MSELoss(reduction='sum')(recon, yb)
			kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
			loss = (recon_loss + beta * kl_div) / xb.size(0)   # per-batch average

			opt.zero_grad()
			loss.backward()
			opt.step()

	return cvae