from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import ranksums
from pymoo.core.termination import Termination

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import Sampling
from pymoo.operators.crossover.hux import HUX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


from joblib import Parallel, delayed

from collections import defaultdict
from joblib import parallel_backend

from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os

class NSGA_II_Filter(Problem):
	n_neighbours = 5
	def __init__(self, X_train, y_train, X_val, y_val):
		self.X_train = np.ascontiguousarray(X_train)
		self.y_train = np.ascontiguousarray(y_train)
		self.X_val, self.y_val = X_val, y_val
		self.n_instances = X_train.shape[0]
		super().__init__(n_var=self.n_instances, n_obj=3, n_constr=0, xl=0, xu=1, type_var=np.bool_)

	def train(self, instance, x1, y1, x2, y2):
		
		values = {
			"Inverse ACC": 1, 
			"Inverse AUC": 1, 
			"Sample Size": 1
		}

		if np.sum(instance) >= NSGA_II_Filter.n_neighbours:
			model = KNeighborsClassifier(n_neighbors=NSGA_II_Filter.n_neighbours)
			model.fit(x1[instance], y1[instance])
			y_pred = model.predict(x2)
			values["Sample Size"] = x1[instance].shape[0] / self.X_train.shape[0]
			values["Inverse AUC"] = 1 - roc_auc_score(y2, y_pred)
			values["Inverse ACC"] = 1 - accuracy_score(y2, y_pred)

		return values
	
	def _evaluate(self, x, out, *args, **kwargs):
		global instance_index
		global do_stop

		with parallel_backend('loky', inner_max_num_threads=1):
		
			results = Parallel(
				n_jobs=-1,
				batch_size=8,               
				mmap_mode='r',              
				prefer='processes'
			)(delayed(self.train)(mask, self.X_train, self.y_train, self.X_val, self.y_val) for mask in x)

		values = []
		for key in results[0]:
			values.append([result[key] for result in results])

		out["F"] = np.column_stack(values)

class DiverseSampling(Sampling):
	def __init__(self):
		super().__init__()
		
	def _do(self, problem, n_samples, **kwargs):
		target_inclusions = np.random.randint(problem.n_var // 3, problem.n_var, n_samples)
		init_pops = []
		for target in target_inclusions:
			array = np.array([1]*target + [0]*(problem.n_var - target))
			np.random.shuffle(array)
			init_pops.append(array)
		init_pops = np.array(init_pops, dtype=np.bool)
	
		return init_pops