import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from matplotlib.cm import get_cmap

def PCA_plot(x, y, cmap='coolwarm'):
	X = np.asarray(x)
	y = np.asarray(y)

	# Standardize features (important for PCA)
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	pca = PCA(n_components=2)
	X_pca = pca.fit_transform(X_scaled)

	plt.figure(figsize=(8,6))
	scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,cmap=cmap, alpha=0.7)
	plt.xlabel('PC 1')
	plt.ylabel('PC 2')
	plt.legend(*scatter.legend_elements(), title="Class")
	plt.grid(True)
	plt.show()

def PCA_plot_rare_on_top(x, y):
	# --- PCA -------------------------------------------------------------
	X_scaled = StandardScaler().fit_transform(x)
	X_pca   = PCA(n_components=2).fit_transform(X_scaled)

	# --- colour palette --------------------------------------------------
	labels         = np.unique(x)
	palette        = get_cmap('coolwarm', len(labels))      # discrete palette
	colour_by_lbl  = {lbl: palette(i) for i, lbl in enumerate(labels)}

	# --- count class frequencies & plot ---------------------------------
	counts         = Counter(y)
	ordered_labels = sorted(counts, key=counts.get, reverse=True)  # rare last

	plt.figure(figsize=(8, 6))
	for z, lbl in enumerate(ordered_labels):
		mask = y == lbl
		plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
					color=colour_by_lbl[lbl], alpha=0.7,
					label=f'{lbl} (n={counts[lbl]})', zorder=z)

	plt.xlabel('PC 1');  plt.ylabel('PC 2');  plt.grid(True)
	plt.legend(title='Class')
	plt.show()