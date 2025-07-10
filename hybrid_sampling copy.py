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

from joblib import Parallel, delayed

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os

from imblearn.over_sampling import (
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
)
from imblearn.combine import (
    SMOTETomek,
    SMOTEENN
)
import wandb


class AUC_Filter(Problem):
	n_neighbours = 5
	log_every = 5
	def __init__(self, X_train, y_train, X_val, y_val, x_test, y_test, inclusion_threshold, logger=None):
		
		self.generation_number = 0

		self.X_train = X_train
		self.y_train = y_train
		self.X_val = X_val
		self.y_val = y_val
		self.X_TEST = x_test
		self.Y_TEST = y_test
		self.logger = logger

		self.high_performers = []
		self.inclusion_threshold = inclusion_threshold

		self.training_data = X_train
		self.n_instances = X_train.shape[0]
		
		super().__init__(
			n_var=self.n_instances,
			# n_obj=1,
			n_obj=2,               
			n_constr=0,            
			xl=0,                  
			xu=1,                  
			type_var=np.bool_,     
		)

	 
	def train(self, instance, x1, y1, x2, y2):
		
		if np.sum(instance) >= AUC_Filter.n_neighbours:
			model = KNeighborsClassifier(
				n_neighbors=AUC_Filter.n_neighbours
			)
			model.fit(
				x1[instance], 
				y1[instance]
			)

			y_pred = model.predict(x2)
			auc = roc_auc_score(y2, y_pred)

			if auc > self.inclusion_threshold:
				self.high_performers.append((x1[instance], y1[instance]))

			return (1 - auc, np.sum(instance))

		return (1, np.sum(instance))
	
	def _evaluate(self, x, out, *args, **kwargs):
		
		results = Parallel(n_jobs=-1)(delayed(self.train)(instance, self.X_train, self.y_train, self.X_val, self.y_val) for instance in x)
		auc = [result[0] for result in results]
		num_samples = [result[1] for result in results]

		F = np.column_stack([auc, num_samples])

		self.generation_number += 1
		max_validation_auc = -1
		real_test_auc = -1
		max_test_auc = -1
		if self.logger is not None and self.generation_number % AUC_Filter.log_every == 0:
			for idx in NonDominatedSorting().do(F, only_non_dominated_front=True):
				instance = x[idx]

				if np.sum(instance) >= AUC_Filter.n_neighbours:
					model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
					model.fit(self.X_train[instance], self.y_train[instance])
					y_pred = model.predict(self.X_val)
					validation_auc = roc_auc_score(self.y_val, y_pred)

					y_pred = model.predict(self.X_TEST)
					test_auc = roc_auc_score(self.Y_TEST, y_pred)
				
					if test_auc > max_test_auc:
						max_test_auc = test_auc
					
					if validation_auc > max_validation_auc:
						max_validation_auc = validation_auc
						real_test_auc = test_auc

			self.logger.log({
				"validation/optimized_AUC": max_validation_auc,
				"test/optimized_AUC": real_test_auc,
				"test/ideal_AUC": max_test_auc,
			})

		out["F"] = F

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

class cVAE:
	def __init__(self, sampling_strategy=None):
		pass

	def fit_resample(self, x, y):
		lr = 1e-3
		epochs = 200
		batch_size = 20
		beta = 0.8
		input_dim = x[0].shape[0]

		minority_label = pd.DataFrame(y).value_counts().argmin()
		minority_indices = np.where(y==minority_label)[0]
		minority_features = x[minority_indices]
		minority_labels = y[minority_indices]

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		train_set = TabularDataset(torch.from_numpy(x), torch.from_numpy(y))
		train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
		cvae = ConditionalVAE(input_dim, 1, input_dim//2, 2).to(device)
		optimizer = optim.Adam(cvae.parameters(), lr=lr)
		cvae.train()

		##############################
		#
		##############################

		for epoch in range(epochs):
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

		#################################
		#
		#################################

		cvae.eval()
		with torch.no_grad():    
			temp_x = torch.from_numpy(minority_features).to(device).float()
			temp_y = torch.from_numpy(minority_labels).to(device).float().unsqueeze(1)
			mu, logvar = cvae.encode(temp_x, temp_y)
			z = cvae.reparameterize(mu, logvar)
			total_latents = z.cpu().numpy()
			variance = np.var(total_latents, axis=0)
	
		################################
		#
		################################

		synthetic_features = []
		num_samples = 50
		while len(synthetic_features) < num_samples:
			with torch.no_grad():    
				temp_x = torch.from_numpy(minority_features).to(device).float()
				temp_y = torch.from_numpy(minority_labels).to(device).float().unsqueeze(1)
				mu, logvar = cvae.encode(temp_x, temp_y)
				z = cvae.reparameterize(mu, logvar)
			minority_latents = z.cpu().numpy()

			for dim in range(2):
				minority_latents[:,dim] += np.random.normal(-variance[dim]/2, variance[dim]/2, len(minority_latents))
			
			with torch.no_grad():    
				z = torch.from_numpy(minority_latents).to(device).float()
				label_dim = torch.from_numpy(minority_labels).to(device).float().unsqueeze(1)
				synthetic_minority_samples = cvae.decode(z, label_dim)
		
			for sample in synthetic_minority_samples.cpu().numpy():
				if len(synthetic_features) < num_samples:
					synthetic_features.append(sample)
				else:
					break
				
		x = np.concatenate((x, np.array(synthetic_features)), axis=0)
		y = np.concatenate((y, [minority_labels[0]] * len(synthetic_features)), axis=0)
		return x, y
	
class DiverseInheritedSampling(Sampling):
	def __init__(self, curr_x_train, prev_samples):
		super().__init__()
		self.inherited_pops = []

		for x, y in prev_samples:
			individual = np.zeros(len(curr_x_train))
			for feature in x:

				idx = None
				for idx, positioned_feature in enumerate(curr_x_train):
					if np.all(positioned_feature == feature):
						break
				else:
					individual[idx] = 1
				
			self.inherited_pops.append(individual)

	def _do(self, problem, n_samples, **kwargs):

		target_inclusions = np.random.randint(
			problem.n_var // 3,
			problem.n_var,
			n_samples-len(self.inherited_pops)
		)
		init_pops = []
		for target in target_inclusions:
			array = np.array([1]*target + [0]*(problem.n_var - target))
			np.random.shuffle(array)
			init_pops.append(array)

		init_pops.extend(self.inherited_pops)
		init_pops = np.array(init_pops, dtype=np.bool)
	
		return init_pops

def execute(data_key, x_train, y_train, x_validation, y_validation, x_test, y_test):
	segments = data_key.split('_')
	dataset_name, split_num = '_'.join(segments[1:]), segments[0]

	logger = None
	logger = wandb.init(
			project="GA Instance Selection Over Sampling", 
			group="hybridSample",
			tags=['2025-07-09-TEST1', dataset_name],		
			name=data_key
		)

	strong_performers = []
	curr_x_train, curr_y_train = x_train, y_train

	model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_validation)
	baseline_AUC = roc_auc_score(y_validation, y_pred)

	max_validation_auc = -1
	real_test_auc = -1
	max_test_auc = -1
	optimized_num_samples = -1
	ideal_num_samles = -1

	print(f"Inital AUC to beat {round(baseline_AUC, 4)}")
	for loop_num in range(3):
		print(f"{data_key} loop num {loop_num}")
		################################################# 
		# Generate the synthetic samples using only training data.
		# While doing so, track the maxima AUC expected
		# pre-optimization to filter which samples are
		# to be inherited to the next iteration.
		#################################################
		carry_over_x, carry_over_y = [], []
		for x, y in strong_performers:
			for features, label in zip(x, y):
				for saved_features in curr_x_train:
					if np.all(features == saved_features): break
				else:
					for prior_feature in carry_over_x:
						if np.all(features == prior_feature): break
					else:
						# If features is a synthetic sample which is not yet 
						# saved, add it to the list of samples
						# being carried over into the GA filter.
						carry_over_x.append(features)
						carry_over_y.append(label)
		
		# If there was something to carry over
		# concatenate it to the real training set 
		combined_curr_x = np.concatenate((curr_x_train, x_validation), axis=0)
		combined_curr_y = np.concatenate((curr_y_train, y_validation), axis=0)

		if carry_over_x != []:
			curr_x_train = np.concatenate((curr_x_train, carry_over_x), axis=0)
			curr_y_train = np.concatenate((curr_y_train, carry_over_y), axis=0)

			combined_curr_x = np.concatenate((combined_curr_x, carry_over_x), axis=0)
			combined_curr_y = np.concatenate((combined_curr_y, carry_over_y), axis=0)

		# Save the index where the previously saved 
		# samples (real and synthetic) begin.
		new_idx = len(combined_curr_x)

		synthetic_features, synthetic_labels = None, None
		for sampler in [SMOTE, ADASYN, BorderlineSMOTE, cVAE, SMOTETomek, SMOTEENN]:
			sampler = sampler()

			try:
				oversample_x, oversample_y = sampler.fit_resample(combined_curr_x, combined_curr_y)
				model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
				model.fit(oversample_x, oversample_y)
				y_pred = model.predict(x_validation)
				auc = roc_auc_score(y_validation, y_pred)

				# If the synthetic set pushes the baseline AUC
				# up, save the new baseline performance
				# this ensures the filter will only save
				# samples from instances who perform better than
				# the baseline.
				# if  auc > baseline_AUC:
				# 	baseline_AUC = auc

				if synthetic_features is not None:
					synthetic_features = np.concatenate((synthetic_features, oversample_x[new_idx:]))
					synthetic_labels = np.concatenate((synthetic_labels, oversample_y[new_idx:]))
				else:
					synthetic_features, synthetic_labels = oversample_x[new_idx:], oversample_y[new_idx:]

			except Exception as e:
				print(f"\tDidnt generate new samples because {e}")

		if logger is not None:
			logger.log({"validation/filter_auc": baseline_AUC})
		
		baseline_AUC = -1
		print(f"Must beat: {baseline_AUC}")

		# Create the candidate training set, with the synthetic features minus the validation
		# set that was used to create the samples.
		candidate_x_train = np.concatenate((curr_x_train, synthetic_features))
		candidate_y_train = np.concatenate((curr_y_train, synthetic_labels))

		#################################################
		# Execute optimization filtering.
		# Given the synthetic samples of SMOTE, ADASYN, and BorderlineSMOTE
		# NSGA-II will attempt to optimize for AUC and number of samples.
		#################################################

		problem = AUC_Filter(
			candidate_x_train, candidate_y_train, 
			x_validation, y_validation,
			x_test, y_test,
			baseline_AUC,
			logger
		)
		algorithm = NSGA2(
			pop_size=500, 
			sampling=DiverseInheritedSampling(candidate_x_train, strong_performers), 
			crossover=HUX(), 
			mutation=BitflipMutation(), 
			eliminate_duplicates=True
		)
		result = minimize(
			problem, 
			algorithm, 
			('n_gen', 100), 
			save_history=False
		)

		#################################################
		# For each individual in the final population track
		# the AUC, save it if the AUC is greater than the baseline 
		# AUC expected with any one of SMOTE, ADAYSN, etc.
		#################################################
		
		samples_auc = []
		candidate_inherited_samples = []
		
		for indidivual in result.pop:
			instance = indidivual.X
			if np.sum(instance) >= AUC_Filter.n_neighbours:

				model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
				model.fit(candidate_x_train[instance], candidate_y_train[instance] )
				y_pred = model.predict(x_validation)
				validation_auc = roc_auc_score(y_validation, y_pred)
				
				# If the current validation AUC beats the pre-optimization best
				# save the sample for later
				if validation_auc > baseline_AUC:
					samples_auc.append(validation_auc)
					candidate_inherited_samples.append((candidate_x_train[instance], candidate_y_train[instance]))

				############################################
				# For logging and final result calculation
				############################################
				y_pred = model.predict(x_test)
				test_auc = roc_auc_score(y_test, y_pred)
				if validation_auc > max_validation_auc:
					max_validation_auc = validation_auc
					real_test_auc = test_auc
					optimized_num_samples = np.sum(instance)
				if test_auc > max_test_auc:
					max_test_auc = test_auc
					ideal_num_samles = np.sum(instance)
				############################################
			else:
				print(f"Error")

		for x, y in result.problem.high_performers:
			model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
			model.fit(x, y)
			y_pred = model.predict(x_validation)
			validation_auc = roc_auc_score(y_validation, y_pred)
			samples_auc.append(validation_auc)
			candidate_inherited_samples.append((x,y))
		# Save the best 20 instances
		# which had performance better
		# than the pre-optimization 
		# set.
		# strong_performers = []
		count, save_count = 0, 20
		indices = np.argsort(samples_auc)
		print(np.array(samples_auc)[indices])
		for idx in indices[::-1]:
			if count > save_count: break
			count += 1
			strong_performers.append(candidate_inherited_samples[idx])

		if logger is not None:
			logger.log({"train/Num strong performers": len(strong_performers)})
		
	record = {
		"Dataset": dataset_name,
		"Split Num": split_num,
		"Optimized validation AUC": max_validation_auc,
		"Optimized test AUC": real_test_auc,
		"Ideal test AUC": max_test_auc,
		"Optimized num samples": optimized_num_samples,
		"Ideal num samples": ideal_num_samles
	}


	if logger is not None:
		logger.finish()

	return record
if __name__ == "__main__":

	with open('data.pickle', 'rb') as fh:
		data_mapper = pickle.load(fh)
	splits = pd.read_csv('data_splits.csv')
	
	records = []
	for dataset in splits.columns:
		for data_key in splits[dataset]:
			try:
				# if "abalone-17_vs_7-8-9-10" in data_key or "abalone19" in data_key: continue

				record = execute(
					data_key, 
					data_mapper[data_key]['x_train'],
					data_mapper[data_key]['y_train'],
					data_mapper[data_key]['x_validation'],
					data_mapper[data_key]['y_validation'],
					data_mapper[data_key]['x_test'],
					data_mapper[data_key]['y_test'],
				)
				records.append(record)
				pd.DataFrame.from_records(records).to_csv("long_run_2025-07-08-IMPROV-PERSIST.csv", index=False)

			except Exception as E:
				print(E)
		break
				
		
