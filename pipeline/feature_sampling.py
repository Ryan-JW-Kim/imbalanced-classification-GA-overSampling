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

def meshgrid_sample_real_feature_space(x, variance_multiplier=2):
	feature_variance = np.var(x, axis=0)
	feature_mins = np.min(x, axis=0)
	feature_maxs = np.max(x, axis=0)
	feature_grids = []
	for idx, var in enumerate(feature_variance):
		lo  = feature_mins[idx] - var      # lower bound  (min – variance)
		hi  = feature_maxs[idx] + var      # upper bound  (max + variance)
		step = var * variance_multiplier or 1e-8             # avoid step == 0 if var == 0
		grid = np.arange(lo, hi + step, step)
		feature_grids.append(grid)

	real_feature_space = np.fromiter((val for combo in product(*feature_grids) for val in combo), dtype=float).reshape(-1, len(feature_grids))

	return real_feature_space