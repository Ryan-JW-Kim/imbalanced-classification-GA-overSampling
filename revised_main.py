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

from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os

import wandb

with open('data.pickle', 'rb') as fh:
	data_mapper = pickle.load(fh)

class AUC_Filter(Problem):
	population_size = 100
	n_neighbours = 5
	log_every = 5
	def __init__(self, X_train, y_train, X_val, y_val, X_test, Y_test, logger=None):
		
		self.generation_number = 0

		self.X_train = X_train
		self.y_train = y_train
		self.X_val = X_val
		self.y_val = y_val
		self.X_TEST = X_test
		self.Y_TEST = Y_test
		self.logger = logger

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
		
		if self.logger is not None and self.generation_number % AUC_Optimizer.log_every == 0:
			validation_aucs = []
			test_aucs = []
			pareto_indices = NonDominatedSorting().do(F, only_non_dominated_front=True)
			
			for idx in pareto_indices:
				instance = x[idx]

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
			})

		out["F"] = F

class AUC_Optimizer(Problem):
	population_size = 100
	n_neighbours = 5
	log_every = 5
	def __init__(self, X_train, y_train, X_val, y_val, X_test, Y_test, logger=None):
		
		self.generation_number = 0

		self.X_train = X_train
		self.y_train = y_train
		self.X_val = X_val
		self.y_val = y_val
		self.X_TEST = X_test
		self.Y_TEST = Y_test
		self.logger = logger

		self.training_data = X_train
		self.n_instances = X_train.shape[0]
		
		super().__init__(
			n_var=self.n_instances,
			n_obj=1,               
			n_constr=0,            
			xl=0,                  
			xu=1,                  
			type_var=np.bool_,     
		)

	def _evaluate(self, x, out, *args, **kwargs):
		
		values = []
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
			
			values.append(inverse_AUC)
		F = np.column_stack([values])
		self.generation_number += 1
		
		if self.logger is not None and self.generation_number % AUC_Optimizer.log_every == 0:
			validation_aucs = []
			test_aucs = []
			pareto_indices = NonDominatedSorting().do(F, only_non_dominated_front=True)
			
			for idx in pareto_indices:
				instance = x[idx]

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

def combine_sets(x1, y1, x2, y2):
	if len(x2) == 0:
		return x1, y1
	
	x = np.concatenate((x1, x2), axis=0)
	y = np.concatenate((y1, y2), axis=0)
	return x, y

def select_synthetic_survivors(result, x_train, y_train, x_validation, y_validation, real_samples, auc_selection_threshold):
	selected_samples = []
	sample_AUCs = []
	for individual in result.pop:

		if np.sum(individual.X) > AUC_Optimizer.n_neighbours:
			model = KNeighborsClassifier(n_neighbors=AUC_Optimizer.n_neighbours)
			model.fit(x_train[individual.X], y_train[individual.X])
			y_pred = model.predict(x_validation)
			optimized_AUC = roc_auc_score(y_validation, y_pred)

			if optimized_AUC > auc_selection_threshold:
				for sample in x_train[individual.X]:
					for real_sample in real_samples:
						if np.all(sample == real_sample):
							break
					else:
						for saved_sample in selected_samples:
							if np.all(sample == saved_sample):
								break
						else:
							sample_AUCs.append(optimized_AUC)
							selected_samples.append(sample)
	
	idx = np.argsort(sample_AUCs)
	top_idx = idx[:10]
	arr = np.array(selected_samples)

	return arr[top_idx]

def split_info(data_key):
	segments = data_key.split('_')
	split_num = segments[0]
	dataset_name = '_'.join(segments[1:])
	return split_num, dataset_name

def calculate_IR(labels):
	counts = pd.DataFrame(labels).value_counts()
	return counts.max()/counts.min() 

def execute(data_key):
	
	split_number, dataset_name = split_info(data_key)
	x_train = data_mapper[data_key]['x_train'] 
	y_train = data_mapper[data_key]['y_train']
	x_validation = data_mapper[data_key]['x_validation'] 
	y_validation = data_mapper[data_key]['y_validation']
	x_test = data_mapper[data_key]['x_test'] 
	y_test = data_mapper[data_key]['y_test']
	
	minority_label = pd.DataFrame(y_train).value_counts().argmin()
	minority_indices = np.where(y_train==minority_label)[0]
	minority_features = x_train[minority_indices]
	minority_labels = y_train[minority_indices]

	input_dim = x_train[0].shape[0]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	cvae = ConditionalVAE(input_dim, 1, input_dim//2, 2).to(device)
	cVAE_x_train, cVAE_y_train = combine_sets(x_train, y_train, x_validation, y_validation)
	cvae = train(cVAE_x_train, cVAE_y_train, cvae, lr=1e-3, epochs=200, batch_size=20, beta=0.8)
	variance = calculate_latent_dimension_variance(minority_features, minority_labels, cvae)
	new_x_train, new_y_train = x_train, y_train

	model = KNeighborsClassifier(n_neighbors=AUC_Optimizer.n_neighbours)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_validation)
	baseline_validation_AUC = roc_auc_score(y_validation, y_pred)
	
	logger = wandb.init(
		project="GA Instance Selection Over Sampling", 
		group="cVAE-GA",
		tags=['2025-06-08', dataset_name],		
		name=data_key
	)

	pass_instances_to_next_iteration = None
	
	for generation_iter in range(2):	
	# for generation_iter in range(5):	

		synthetic_minority_features = generate_synthetic_examples(minority_features, minority_labels, variance, cvae)
		candidate_x_train, candidate_y_train = combine_sets(new_x_train, new_y_train, synthetic_minority_features, [minority_labels[0]] * len(synthetic_minority_features))
		
		problem = AUC_Filter(
			candidate_x_train, candidate_y_train, 
			x_validation, y_validation,
			X_test=x_test, Y_test=y_test,
			logger=logger
		)
		algorithm = NSGA2(pop_size=AUC_Optimizer.population_size, sampling=DiverseCustomSampling(), crossover=HUX(), mutation=BitflipMutation(), eliminate_duplicates=True)
		result = minimize(problem, algorithm, ('n_gen', AUC_Optimizer.population_size), save_history=False)
		
		pass_instances_to_next_iteration = find_top_AUC_from_result(result, candidate_x_train)
		
		
		synthetic_survivors = select_synthetic_survivors(
			result, 
			candidate_x_train, candidate_y_train,
			x_validation, y_validation,
			new_x_train,
			baseline_validation_AUC,
		)

		# with open(f'results/{data_key} iter{generation_iter}.pickle', 'wb') as fh:
		# 	pickle.dump(
		# 		{
		# 			"Base x_train": candidate_x_train,
		# 			"Base y_train": candidate_y_train,
		# 			"Contrastive x_train": new_x_train,
		# 			"Result": result
		# 		},
		# 		fh
		# 	)
		new_x_train, new_y_train = combine_sets(new_x_train, new_y_train, synthetic_survivors, [minority_labels[0]] * len(synthetic_survivors))

		# Retrain cVAE upon these samples.
		cVAE_x_train, cVAE_y_train = combine_sets(new_x_train, new_y_train, x_validation, y_validation)
		cvae = ConditionalVAE(input_dim, 1, input_dim//2, 2).to(device)
		cvae = train(new_x_train, new_y_train, cvae, lr=1e-3, epochs=200, batch_size=20, beta=0.8, logger=logger)

		logger.log({
			"surviving samples": len(synthetic_survivors),
			"train/size": len(new_x_train),
			"train/IR": calculate_IR(new_y_train)
		})
	
	logger.finish()
	return synthetic_survivors

def baseline_run(data_key, scheme_key):
	
	split_number, dataset_name = split_info(data_key)

	x_train = data_mapper[data_key]['x_train'] 
	y_train = data_mapper[data_key]['y_train']
	x_validation = data_mapper[data_key]['x_validation'] 
	y_validation = data_mapper[data_key]['y_validation']
	x_test = data_mapper[data_key]['x_test'] 
	y_test = data_mapper[data_key]['y_test']
	
	ga_logger = wandb.init(
		project="GA Instance Selection Over Sampling", 
		group=f"{scheme_key}:{dataset_name}",
		name=data_key
	)
	problem = AUC_Optimizer(
		x_train, y_train, 
		x_validation, y_validation,
		X_test=x_test,
		Y_test=y_test,
		logger=ga_logger
	)
	algorithm = NSGA2(pop_size=AUC_Optimizer.population_size, sampling=DiverseCustomSampling(), crossover=HUX(), mutation=BitflipMutation(), eliminate_duplicates=True)
	result = minimize(problem, algorithm, ('n_gen', AUC_Optimizer.population_size), save_history=False)
	ga_logger.finish()

	return result
	
if __name__ == "__main__":
	splits = pd.read_csv('data_splits.csv')

	print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print(F"Executing for:")

	# for dataset in splits.columns[:len(splits.columns)//2]:
	for dataset in splits.columns[len(splits.columns)//2:]:
		for data_key in splits[dataset]:
			try:
				synthetic_samples = execute(data_key)
				print(data_key)
			except Exception as e:
				print(f"Error at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {e}")	
			break
		break