from torch.utils.data import Dataset
import torch

class TabularDataset(Dataset):
	def __init__(self, x_synthetic, x_true, shared_label):
		self.x = x_synthetic
		self.y = x_true
		self.label = shared_label
	def __len__(self):
		return self.x.shape[0]
	def __getitem__(self, ind):
		return self.x[ind], self.y[ind], self.label[ind]
	
def mask_features(x, min_mask=1, max_mask=3):
	x_masked = x.clone()
	for i in range(x.size(0)):
		mask_count = torch.randint(min_mask, max_mask + 1, (1,)).item()
		mask_indices = torch.randperm(x.size(1))[:mask_count]
		x_masked[i, mask_indices] = 0
	return x_masked