from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score, roc_auc_score

from pymoo.operators.mutation.bitflip import BitflipMutation, Mutation
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.sampling.rnd import BinaryRandomSampling, Sampling
from pymoo.operators.crossover.hux import HUX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import Hypervolume
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch

from scipy.stats import ranksums

from joblib import Parallel, delayed
from pathlib import Path
from io import StringIO
import pandas as pd
import numpy as np
import time
import pickle
import os
import re
from datetime import datetime

import wandb

import matplotlib.pyplot as plt

with open('data.pickle', 'rb') as fh:
	data_mapper = pickle.load(fh)

class AUC_Optimizer(Problem):
	population_size = 100
	n_neighbours = 5
	log_every = 5
	def __init__(self, X_train, y_train, X_val, y_val, X_test, Y_test, synthetic_samples, logger=None):
		
		self.generation_number = 0

		self.X_train = X_train
		self.y_train = y_train
		self.X_val = X_val
		self.y_val = y_val
		self.X_TEST = X_test
		self.Y_TEST = Y_test
		self.logger = logger
		self.synthetic_samples = synthetic_samples

		self.training_data = X_train
		self.n_instances = X_train.shape[0]
		
		super().__init__(
			n_var=self.n_instances,
			n_obj=2,               
			n_constr=0,            
			xl=0,                  
			xu=1,                  
			type_var=np.bool_,     
		)

	def _evaluate(self, x, out, *args, **kwargs):
		
		values = []
		num_samples = []
		for instance in x:
			inverse_AUC = 1
			if np.sum(instance) >= AUC_Optimizer.n_neighbours:
				model = KNeighborsClassifier(
					n_neighbors=AUC_Optimizer.n_neighbours
				)
				model.fit(
					self.X_train[instance], 
					self.y_train[instance]
				)

				y_pred = model.predict(self.X_val)
				inverse_AUC = 1 - roc_auc_score(self.y_val, y_pred)
			
			num_samples.append(np.sum(instance))
			values.append(inverse_AUC)
		F = np.column_stack([values, num_samples])
		self.generation_number += 1
		
		if self.logger is not None and self.generate_synthetic_examples % AUC_Optimizer.log_every == 0:
			validation_aucs = []
			test_aucs = []
			pareto_indices = NonDominatedSorting().do(F, only_non_dominated_front=True)
			synthetic_pareto_samples = 0
			
			for idx in pareto_indices:
				instance = x[idx]
				for sample in self.X_train[instance]:
					for synthetic_sample in self.synthetic_samples:
						if np.all(sample == synthetic_sample):
							synthetic_pareto_samples += 1
							break

				if np.sum(instance) >= AUC_Optimizer.n_neighbours:
					model = KNeighborsClassifier(n_neighbors=AUC_Optimizer.n_neighbours)
					model.fit(
						self.X_train[instance], 
						self.y_train[instance]
					)
					y_pred = model.predict(self.X_val)
					validation_aucs.append(roc_auc_score(self.y_val, y_pred))
					y_pred = model.predict(self.X_TEST)
					test_aucs.append(roc_auc_score(self.Y_TEST, y_pred))
				else:
					validation_aucs.append(0)
					test_aucs.append(0)
					
			validation_idx = np.argmax(validation_aucs)
			test_idx = np.argmax(test_aucs)	
			x_ideal_validation_instance = self.X_train[x[pareto_indices[validation_idx]]]
			y_ideal_validation_instance = self.y_train[x[pareto_indices[validation_idx]]]
			x_ideal_test_instance = self.X_train[x[pareto_indices[test_idx]]]
			y_ideal_test_instance = self.y_train[x[pareto_indices[test_idx]]]

			if len(x_ideal_validation_instance) >= AUC_Optimizer.n_neighbours:
				model = KNeighborsClassifier(n_neighbors=AUC_Optimizer.n_neighbours)
				model.fit(
					x_ideal_validation_instance, 
					y_ideal_validation_instance
				)

				y_pred = model.predict(self.X_val)
				optimized_validation_auc = roc_auc_score(self.y_val, y_pred)
				
				y_pred = model.predict(self.X_TEST)
				optimized_test_auc = roc_auc_score(self.Y_TEST, y_pred)
			else:
				optimized_validation_auc = 0
				optimized_test_auc = 0

			# Calculate metrics using ideal instance w.r.t test AUC
			if len(x_ideal_test_instance) >= AUC_Optimizer.n_neighbours:
				model = KNeighborsClassifier(n_neighbors=AUC_Optimizer.n_neighbours)
				model.fit(
					x_ideal_test_instance, 
					y_ideal_test_instance
				)
				
				y_pred = model.predict(self.X_TEST)
				ideal_test_auc = roc_auc_score(self.Y_TEST, y_pred)
			else:
				ideal_test_auc

			self.logger.log({
				"validation/optimized_AUC": optimized_validation_auc,
				"test/optimized_AUC": optimized_test_auc,
				"test/ideal_AUC": ideal_test_auc,
				"train/synthetic_pareto_samples": synthetic_pareto_samples
			})

		out["F"] = F
			
class DiverseCustomSampling(Sampling):
	def __init__(self):
		super().__init__()

	def _do(self, problem, n_samples, **kwargs):

		target_inclusions = np.random.randint(
			problem.n_var // 3,
			problem.n_var,
			n_samples
		)
		init_pops = []
		for target in target_inclusions:
			array = np.array([1]*target + [0]*(problem.n_var - target))
			np.random.shuffle(array)
			init_pops.append(array)
		init_pops = np.array(init_pops, dtype=np.bool)
		return init_pops

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

class CustomDataset(Dataset):
	def __init__(self, x_synthetic, x_true):
		self.x = x_synthetic
		self.y = x_true
	def __len__(self):
		return self.x.shape[0]
	def __getitem__(self, ind):
		x = self.x[ind]
		y = self.y[ind]
		return x, y

def train(training_x, training_y, cvae, lr, epochs, batch_size, beta, logger=None):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_set = CustomDataset(torch.from_numpy(training_x), torch.from_numpy(training_y))
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	optimizer = optim.Adam(cvae.parameters(), lr=lr)
	
	for epoch in range(epochs):
		cvae.train()
		total_loss = 0
		for batch in train_loader:
			x_batch = batch[0].to(device).float()
			y_batch = batch[1].to(device).float().unsqueeze(1)
			
			recon, mu, logvar = cvae(x_batch, y_batch)
			recon_loss = nn.MSELoss()(recon, x_batch)
			kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
			loss = recon_loss + (kl_div*beta)
	
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			total_loss += loss.item()
		if logger is not None:
			logger.log({
				"training loss": total_loss / len(train_loader)
			})
		
	return cvae

def generate_synthetic_examples(x_samples, y_samples, sample_variance, cvae, num_samples=None):
	if num_samples is None:
		num_samples = len(x_samples)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	cvae.eval()

	synthetic_features = []

	while len(synthetic_features) < num_samples:
		with torch.no_grad():    
			x = torch.from_numpy(x_samples).to(device).float()
			y = torch.from_numpy(y_samples).to(device).float().unsqueeze(1)
			mu, logvar = cvae.encode(x, y)
			z = cvae.reparameterize(mu, logvar)
		minority_latents = z.cpu().numpy()

		minority_latents[:,0] += np.random.normal(-sample_variance[0]/2, sample_variance[0]/2, len(minority_latents))
		minority_latents[:,1] += np.random.normal(-sample_variance[1]/2, sample_variance[1]/2, len(minority_latents))

		with torch.no_grad():    
			z = torch.from_numpy(minority_latents).to(device).float()
			label_dim = torch.from_numpy(y_samples).to(device).float().unsqueeze(1)
			synthetic_minority_samples = cvae.decode(z, label_dim)
	
		for sample in synthetic_minority_samples.cpu().numpy():
			if len(synthetic_features) < num_samples:
				synthetic_features.append(sample)
			else:
				break
			

	return np.array(synthetic_features)

def calculate_latent_dimension_variance(x, y, cvae):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	with torch.no_grad():    
		x = torch.from_numpy(x).to(device).float()
		y = torch.from_numpy(y).to(device).float().unsqueeze(1)
		mu, logvar = cvae.encode(x, y)
		z = cvae.reparameterize(mu, logvar)
		total_latents = z.cpu().numpy()
		variance = np.var(total_latents, axis=0)
	
	return variance

def add_synthetic_data(x_train, y_train, x_synthetic, y_synthetic):
	x = np.concatenate((x_train, x_synthetic), axis=0)
	y = np.concatenate((y_train, y_synthetic), axis=0)
	return x, y

# def select_synthetic_survivors(result, x, synthetic_features):
# 	all_samples = []
# 	for instance in result.X:
# 		for sample in x[instance]:
# 			for stored_sample in all_samples:
# 				if np.all(sample == stored_sample):
# 					break
# 			else:
# 				all_samples.append(sample)

# 	# If any of the individuals contain a sample which is synthetic, add it to the
# 	# 'proven' synthetic sample list
# 	synthesized_features = []
# 	for sample in all_samples:
# 		for synthetic_sample in synthetic_features:
# 			if np.all(sample == synthetic_sample):
# 				synthesized_features.append(synthetic_sample)
# 				break
# 	return synthesized_features

def select_synthetic_survivors(result, x, real_samples):
	all_samples = []
	for instance in result.X:
		for sample in x[instance]:
			for stored_sample in all_samples:
				if np.all(sample == stored_sample):
					break
			else:
				all_samples.append(sample)

	synthesized_features = []
	for sample in all_samples:
		for real_sample in real_samples:
			if np.all(sample == real_sample):
				break
		else:
			synthesized_features.append(sample)
			
	return synthesized_features


def split_info(data_key):
	segments = data_key.split('_')
	split_num = segments[0]
	dataset_name = '_'.join(segments[1:])
	return split_num, dataset_name

def calculate_IR(labels):
	counts = pd.DataFrame(labels).value_counts()
	return counts.max()/counts.min() 

def execute_1(data_key, scheme_key):
	
	split_number, dataset_name = split_info(data_key)

	x_train = data_mapper[data_key]['x_train'] 
	y_train = data_mapper[data_key]['y_train']
	x_validation = data_mapper[data_key]['x_validation'] 
	y_validation = data_mapper[data_key]['y_validation']
	x_test = data_mapper[data_key]['x_test'] 
	y_test = data_mapper[data_key]['y_test']
	
	# Define configuration of cVAE
	input_dim = x_train[0].shape[0]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	cvae = ConditionalVAE(input_dim, 1, input_dim//2, 2).to(device)

	# Train cVAE - just real training data
	cvae = train(x_train, y_train, cvae, lr=1e-3, epochs=200, batch_size=20, beta=0.8)

	# Calculate the variance of all samples (minority and majority) within the training data
	variance = calculate_latent_dimension_variance(x_train, y_train, cvae)

	# Extract the data of minority class samples
	minority_label = pd.DataFrame(y_train).value_counts().argmin()
	minority_indices = np.where(y_train==minority_label)[0]
	minority_features = x_train[minority_indices]
	minority_labels = y_train[minority_indices]

	# Initial states
	all_surviving_samples = np.array([[None] * x_train.shape[1]])
	first = True
	new_x_train = x_train
	new_y_train = y_train
	
	outside_loop = wandb.init(
		project="GA Instance Selection Over Sampling", 
		group=f"{scheme_key}:{dataset_name}",
		name=data_key
	)

	for generation_iter in range(20):		
		# Generate minority samples using cVAE latent space and +/- (variance/2)
		synthetic_minority_features = generate_synthetic_examples(minority_features, minority_labels, variance, cvae, num_samples=100)
		new_x_train, new_y_train = add_synthetic_data(new_x_train, new_y_train, synthetic_minority_features, [minority_labels[0]] * len(synthetic_minority_features))
		
		# Optimize to filter low quality synthetic samples
		ga_sampleFiltering = wandb.init(
			project="GA Instance Selection Over Sampling", 
			group=f"{scheme_key}:{generation_iter}:{dataset_name}",
			name=data_key
		)
		problem = AUC_Optimizer(
			new_x_train, new_y_train, 
			x_validation, y_validation,
			X_test=x_test,
			Y_test=y_test,
			synthetic_samples=np.concatenate((all_surviving_samples, synthetic_minority_features)),
			logger=ga_sampleFiltering
		)
		algorithm = NSGA2(pop_size=AUC_Optimizer.population_size, sampling=DiverseCustomSampling(), crossover=HUX(), mutation=BitflipMutation(), eliminate_duplicates=True)
		result = minimize(problem, algorithm, ('n_gen', AUC_Optimizer.population_size), save_history=False)
		
		# For each individual within the final pareto front
		synthetic_survivors = select_synthetic_survivors(result, new_x_train, synthetic_minority_features)

		if first and len(synthetic_survivors) != 0:
			all_surviving_samples = np.array(synthetic_survivors)
			first = False

		elif not first and len(synthetic_survivors) != 0:
			all_surviving_samples = np.concatenate((all_surviving_samples, synthetic_survivors))

		new_x_train, new_y_train = add_synthetic_data(x_train, y_train, all_surviving_samples, [minority_labels[0]] * len(all_surviving_samples))
		
		# new_x_train, new_y_train = add_synthetic_data(x_train, y_train, synthetic_survivors, [minority_labels[0]] * len(all_surviving_samples))

		# Retrain cVAE upon these samples.
		cVAE_training = wandb.init(
			project="GA Instance Selection Over Sampling", 
			group=f"{scheme_key}:{generation_iter}:{dataset_name}",
			name=data_key
		)
		cvae = ConditionalVAE(input_dim, 1, input_dim//2, 2).to(device)
		cvae = train(new_x_train, new_y_train, cvae, lr=1e-3, epochs=200, batch_size=20, beta=0.8, logger=cVAE_training)

		outside_loop.log({
			"samples saved": len(all_surviving_samples),
			"surviving samples": len(synthetic_survivors),
			"train/size": len(new_x_train),
			"train/IR": calculate_IR(new_y_train)
		})

	outside_loop.finish()
	return all_surviving_samples

def execute_2(data_key, scheme_key):
	
	split_number, dataset_name = split_info(data_key)

	x_train = data_mapper[data_key]['x_train'] 
	y_train = data_mapper[data_key]['y_train']
	x_validation = data_mapper[data_key]['x_validation'] 
	y_validation = data_mapper[data_key]['y_validation']
	x_test = data_mapper[data_key]['x_test'] 
	y_test = data_mapper[data_key]['y_test']
	
	# Define configuration of cVAE
	input_dim = x_train[0].shape[0]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	cvae = ConditionalVAE(input_dim, 1, input_dim//2, 2).to(device)

	# Train cVAE - just real training data
	cvae = train(x_train, y_train, cvae, lr=1e-3, epochs=200, batch_size=20, beta=1)

	# Calculate the variance of all samples (minority and majority) within the training data
	variance = calculate_latent_dimension_variance(x_train, y_train, cvae)

	# Extract the data of minority class samples
	minority_label = pd.DataFrame(y_train).value_counts().argmin()
	minority_indices = np.where(y_train==minority_label)[0]
	minority_features = x_train[minority_indices]
	minority_labels = y_train[minority_indices]

	# Initial states
	all_surviving_samples = np.array([[None] * x_train.shape[1]])
	first = True
	new_x_train = x_train
	new_y_train = y_train
	
	outside_loop = wandb.init(
		project="GA Instance Selection Over Sampling", 
		group=f"{scheme_key}:{dataset_name}",
		name=data_key
	)

	for generation_iter in range(5):		
		# Generate minority samples using cVAE latent space and +/- (variance/2)
		synthetic_minority_features = generate_synthetic_examples(minority_features, minority_labels, variance, cvae, num_samples=100)
		new_x_train, new_y_train = add_synthetic_data(new_x_train, new_y_train, synthetic_minority_features, [minority_labels[0]] * len(synthetic_minority_features))
		
		# Optimize to filter low quality synthetic samples
		ga_sampleFiltering = wandb.init(
			project="GA Instance Selection Over Sampling", 
			group=f"{scheme_key}:{generation_iter}:{dataset_name}",
			name=data_key
		)
		problem = AUC_Optimizer(
			new_x_train, new_y_train, 
			x_validation, y_validation,
			X_test=x_test,
			Y_test=y_test,
			synthetic_samples=np.concatenate((all_surviving_samples, synthetic_minority_features)),
			logger=ga_sampleFiltering
		)
		algorithm = NSGA2(pop_size=AUC_Optimizer.population_size, sampling=DiverseCustomSampling(), crossover=HUX(), mutation=BitflipMutation(), eliminate_duplicates=True)
		result = minimize(problem, algorithm, ('n_gen', AUC_Optimizer.population_size), save_history=False)
		
		# For each individual within the final pareto front
		synthetic_survivors = select_synthetic_survivors(result, new_x_train, synthetic_minority_features)

		if first and len(synthetic_survivors) != 0:
			all_surviving_samples = np.array(synthetic_survivors)
			first = False

		elif not first and len(synthetic_survivors) != 0:
			all_surviving_samples = np.concatenate((all_surviving_samples, synthetic_survivors))

		new_x_train, new_y_train = add_synthetic_data(x_train, y_train, all_surviving_samples, [minority_labels[0]] * len(all_surviving_samples))
		
		# new_x_train, new_y_train = add_synthetic_data(x_train, y_train, synthetic_survivors, [minority_labels[0]] * len(all_surviving_samples))

		# Retrain cVAE upon these samples.
		cVAE_training = wandb.init(
			project="GA Instance Selection Over Sampling", 
			group=f"{scheme_key}:{generation_iter}:{dataset_name}",
			name=data_key
		)
		cvae = ConditionalVAE(input_dim, 1, input_dim//2, 2).to(device)
		cvae = train(new_x_train, new_y_train, cvae, lr=1e-3, epochs=200, batch_size=20, beta=1, logger=cVAE_training)

		outside_loop.log({
			"samples saved": len(all_surviving_samples),
			"surviving samples": len(synthetic_survivors),
			"train/size": len(new_x_train),
			"train/IR": calculate_IR(new_y_train)
		})

	outside_loop.finish()
	return all_surviving_samples

def execute_3(data_key, scheme_key):
	
	split_number, dataset_name = split_info(data_key)

	x_train = data_mapper[data_key]['x_train'] 
	y_train = data_mapper[data_key]['y_train']
	x_validation = data_mapper[data_key]['x_validation'] 
	y_validation = data_mapper[data_key]['y_validation']
	x_test = data_mapper[data_key]['x_test'] 
	y_test = data_mapper[data_key]['y_test']
	
	# Define configuration of cVAE
	input_dim = x_train[0].shape[0]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	cvae = ConditionalVAE(input_dim, 1, input_dim//2, 2).to(device)
	cvae = train(x_train, y_train, cvae, lr=1e-3, epochs=200, batch_size=20, beta=0.8)
	variance = calculate_latent_dimension_variance(x_train, y_train, cvae)

	# Extract the data of minority class samples
	minority_label = pd.DataFrame(y_train).value_counts().argmin()
	minority_indices = np.where(y_train==minority_label)[0]
	minority_features = x_train[minority_indices]
	minority_labels = y_train[minority_indices]

	# Initial states
	new_x_train = x_train
	new_y_train = y_train
	
	outside_loop = wandb.init(
		project="GA Instance Selection Over Sampling", 
		group=f"{scheme_key}:{dataset_name}",
		name=data_key
	)

	for generation_iter in range(20):		
		synthetic_minority_features = generate_synthetic_examples(minority_features, minority_labels, variance, cvae, num_samples=100)
		
		new_x_train, new_y_train = add_synthetic_data(new_x_train, new_y_train, synthetic_minority_features, [minority_labels[0]] * len(synthetic_minority_features))
		
		# Optimize to filter low quality synthetic samples
		ga_sampleFiltering = wandb.init(
			project="GA Instance Selection Over Sampling", 
			group=f"{scheme_key}:{generation_iter}:{dataset_name}",
			name=data_key
		)
		problem = AUC_Optimizer(
			new_x_train, new_y_train, 
			x_validation, y_validation,
			X_test=x_test,
			Y_test=y_test,
			synthetic_samples=np.concatenate((all_surviving_samples, synthetic_minority_features)),
			logger=ga_sampleFiltering
		)
		algorithm = NSGA2(pop_size=AUC_Optimizer.population_size, sampling=DiverseCustomSampling(), crossover=HUX(), mutation=BitflipMutation(), eliminate_duplicates=True)
		result = minimize(problem, algorithm, ('n_gen', AUC_Optimizer.population_size), save_history=False)
		
		# For each individual within the final pareto front
		synthetic_survivors = select_synthetic_survivors(result, new_x_train, x_train)
		
		new_x_train, new_y_train = add_synthetic_data(x_train, y_train, synthetic_survivors, [minority_labels[0]] * len(synthetic_survivors))

		# Retrain cVAE upon these samples.
		cVAE_training = wandb.init(
			project="GA Instance Selection Over Sampling", 
			group=f"{scheme_key}:{generation_iter}:{dataset_name}",
			name=data_key
		)
		cvae = train(new_x_train, new_y_train, cvae, lr=1e-3, epochs=200, batch_size=20, beta=0.8, logger=cVAE_training)

		outside_loop.log({
			"samples saved": len(all_surviving_samples),
			"surviving samples": len(synthetic_survivors),
			"train/size": len(new_x_train),
			"train/IR": calculate_IR(new_y_train)
		})

	outside_loop.finish()
	return all_surviving_samples

if __name__ == "__main__":
	splits = pd.read_csv('data_splits.csv')

	print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	for data_key in splits['yeast4']:
		# try:		
		print(f"Start {data_key} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
		if os.path.exists(f"results/{data_key}.result"): continue
		synthetic_samples = execute_1(data_key,"0_8_longrun-non-persist")		
		# synthetic_samples = execute_2(data_key, "1_0")		
		
		# pd.DataFrame(synthetic_samples).to_csv(f'results/{data_key}.csv', index=False)
		# print(f"\tDone {data_key} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

		# except Exception as e:
		# 	print(f"Error at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {e}")