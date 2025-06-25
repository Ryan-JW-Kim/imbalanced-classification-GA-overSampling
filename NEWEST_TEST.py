from imblearn.over_sampling import (
	SMOTE,
	ADASYN,
	BorderlineSMOTE,
)
from imblearn.combine import (
	SMOTETomek,
	SMOTEENN
)

from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

import wandb

def execute(data_key):
	pass

def main():
	splits = pd.read_csv('data_splits.csv')
	
	with open('data.pickle', 'rb') as fh:
		data_mapper = pickle.load(fh)
	

	for dataset in splits.columns[len(splits.columns)//2:]:
		for data_key in splits[dataset]:
			try:
				result = execute(data_key)
				print(f"Done: {data_key}")
				
			except Exception as e:
				print(f"Error at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {e}")	
			break
		break

if __name__ == "__main__":
	main()