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
	def __init__(self, X_train, y_train, X_val, y_val, x_test, y_test, logger=None):
		
		self.generation_number = 0

		self.X_train = X_train
		self.y_train = y_train
		self.X_val = X_val
		self.y_val = y_val
		self.X_TEST = x_test
		self.Y_TEST = y_test
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

class DiverseInheritedSampling(Sampling):
	def __init__(self, curr_x_train, prev_samples):
		super().__init__()

		self.inherited_pops = []

		for x, y in prev_samples:
			individual = np.zeros(len(curr_x_train))
			for feature in x:
				idx = np.argwhere(curr_x_train==feature)		
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
	split_num = segments[0]
	dataset_name = '_'.join(segments[1:])

	logger = wandb.init(
			project="GA Instance Selection Over Sampling", 
			group="hybridSample",
			tags=['2025-06-27', dataset_name],		
			name=data_key
		)
	strong_performers = []
	curr_x_train = x_train
	curr_y_train = y_train
	for _ in range(5):
		print(f"\tIter{_}")

		################################################# 
		# Generate the synthetic samples using only training data.
		# While doing so, track the maxima AUC expected
		# pre-optimization to filter which samples are
		# to be inherited to the next iteration.
		#################################################

		baseline_AUC = -1
		synthetic_features, synthetic_labels = None, None
		
		carry_over_x = []
		carry_over_y = []
		for x, y in strong_performers:
			for feature, label in zip(x, y):
				
				for real_feature in x_train:
					if np.all(feature == real_feature):
						break
				else:
					for prior_feature in carry_over_x:
						if np.all(feature == prior_feature):
							break
					else:
						carry_over_x.append(feature)
						carry_over_y.append(label)

		if carry_over_x != []:
			curr_x_train = np.concatenate((x_train, carry_over_x), axis=0)
			curr_y_train = np.concatenate((y_train, carry_over_y), axis=0)

		new_idx = len(curr_x_train)

		for sampler in [SMOTE, ADASYN, BorderlineSMOTE]:
		
			sampler = sampler(sampling_strategy='minority')
			oversample_x, oversample_y = sampler.fit_resample(curr_x_train, curr_y_train)

			model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
			model.fit(oversample_x, oversample_y)
			y_pred = model.predict(x_validation)
			auc = roc_auc_score(y_validation, y_pred)
			if  auc > baseline_AUC:
				baseline_AUC = auc

			if synthetic_features is not None:
				synthetic_features = np.concatenate((synthetic_features, oversample_x[new_idx:]))
				synthetic_labels = np.concatenate((synthetic_labels, oversample_y[new_idx:]))
			else:
				synthetic_features = oversample_x[new_idx:]
				synthetic_labels = oversample_y[new_idx:]

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
			logger
		)
		algorithm = NSGA2(
			pop_size=AUC_Filter.population_size, 
			sampling=DiverseInheritedSampling(candidate_x_train, strong_performers), 
			crossover=HUX(), 
			mutation=BitflipMutation(), 
			eliminate_duplicates=True
		)
		result = minimize(
			problem, 
			algorithm, 
			('n_gen', AUC_Filter.population_size), 
			save_history=False
		)

		#################################################
		# For each individual in the final population track
		# the AUC, save it if the AUC is greater than the baseline 
		# AUC expected with any one of SMOTE, ADAYSN, etc.
		#################################################

		strong_performers = []
		# print(result.pop)
		for indidivual in result.pop:
			instance = indidivual.X
			if np.sum(instance) >= AUC_Filter.n_neighbours:
				model = KNeighborsClassifier(n_neighbors=AUC_Filter.n_neighbours)
				model.fit(candidate_x_train[instance], candidate_y_train[instance] )
				y_pred = model.predict(x_validation)
				if roc_auc_score(y_validation, y_pred) > baseline_AUC:
					strong_performers.append((
						candidate_x_train[instance],
						candidate_y_train[instance]
					))

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
