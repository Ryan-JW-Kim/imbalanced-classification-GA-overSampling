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

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN


import wandb


class AUC_Filter(Problem):
	population_size = 100
	n_neighbours = 5
	log_every = 5
	def __init__(self, X_train, y_train, X_val, y_val, logger=None):
		
		self.generation_number = 0

		self.X_train = X_train
		self.y_train = y_train
		self.X_val = X_val
		self.y_val = y_val
		# self.X_TEST = X_test
		# self.Y_TEST = Y_test
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
			if np.sum(instance) >= AUC_Filter.n_neighbours:
				model = KNeighborsClassifier(
					n_neighbors=AUC_Filter.n_neighbours
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
		
		if self.logger is not None and self.generation_number % AUC_Filter.log_every == 0:
			validation_aucs = []
			test_aucs = []
			pareto_indices = NonDominatedSorting().do(F, only_non_dominated_front=True)
			
			for idx in pareto_indices:
				instance = x[idx]

				if np.sum(instance) >= AUC_Filter.n_neighbours:
					model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
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

			if len(x_ideal_validation_instance) >= AUC_Filter.n_neighbours:
				model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
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
			if len(x_ideal_test_instance) >= AUC_Filter.n_neighbours:
				model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
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

def execute(data_key, x_train, y_train, x_validation, y_validation, WARNING_TEST_X, WARNING_TEST_Y):
	print(f">{x_train.shape}")
	print(f"{pd.DataFrame(y_train).value_counts()}")
	# for idx in range(len(x_train)):
	# 	print(x_train[idx,:].shape)
	
	synthetic_features = None
	synthetic_labels = None
	real_idx = len(x_train)
	for sampler in [SMOTE, ADASYN, BorderlineSMOTE]:

		sampler = sampler(sampling_strategy='minority')
		oversample_x, oversample_y = sampler.fit_resample(x_train, y_train)

		model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
		model.fit(oversample_x, oversample_y)
		y_pred = model.predict(x_validation)
		print(f"{sampler} AUC: {roc_auc_score(y_validation, y_pred)}")

		if synthetic_features is not None:
			synthetic_features = np.concatenate((synthetic_features, oversample_x[real_idx:]))
			synthetic_labels = np.concatenate((synthetic_labels, oversample_y[real_idx:]))
		else:
			synthetic_features = oversample_x[real_idx:]
			synthetic_labels = oversample_y[real_idx:]

	candidate_x_train = np.concatenate((x_train, synthetic_features))
	candidate_y_train = np.concatenate((y_train, synthetic_labels))

	problem = AUC_Filter(
		candidate_x_train, candidate_y_train, 
		x_validation, y_validation,
	)
	algorithm = NSGA2(pop_size=AUC_Filter.population_size, sampling=DiverseCustomSampling(), crossover=HUX(), mutation=BitflipMutation(), eliminate_duplicates=True)
	result = minimize(problem, algorithm, ('n_gen', AUC_Filter.population_size), save_history=False)
	
	validation_auc = []
	for instance in result.X:
		if np.sum(instance) >= AUC_Filter.n_neighbours:
			model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
			model.fit(candidate_x_train[instance], candidate_y_train[instance] )
			y_pred = model.predict(x_validation)
			validation_auc.append(roc_auc_score(y_validation, y_pred))
		else:

			validation_auc.append(0)

	model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
	model.fit(x_train, y_train)
	y_pred = model.predict(WARNING_TEST_X)
	print(f"Baseline test AUC: {roc_auc_score(WARNING_TEST_Y, y_pred)}")
	
	print(f"Optimized validation AUC: {np.max(validation_auc)}")
	best_idx = np.argmax(validation_auc)
	validation_best_instance = result.X[best_idx]
	model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)

	model.fit(candidate_x_train[validation_best_instance], candidate_y_train[validation_best_instance] )
	y_pred = model.predict(WARNING_TEST_X)
	print(f"Optimized test AUC: {roc_auc_score(WARNING_TEST_Y, y_pred)}")
	
	print(synthetic_features.shape)

if __name__ == "__main__":

	with open('data.pickle', 'rb') as fh:
		data_mapper = pickle.load(fh)
	splits = pd.read_csv('data_splits.csv')


	for dataset in splits.columns:
		for data_key in splits[dataset]:
			print(data_key)
			execute(
				data_key, 
				data_mapper[data_key]['x_train'],
				data_mapper[data_key]['y_train'],
				data_mapper[data_key]['x_validation'],
				data_mapper[data_key]['y_validation'],
				data_mapper[data_key]['x_test'],
				data_mapper[data_key]['y_test'],
			)

			break
		break