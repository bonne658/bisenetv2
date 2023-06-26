import os, sys, cv2
import numpy as np
import torch, glob
from torch.utils.data import Dataset

class LWDDataset(Dataset):
	def __init__(self, src_dir, train=True):
		self.jpgs = glob.glob(src_dir+'/*jpg')
		self.jpgs.sort()
		self.train = train
	def __len__(self):
		return len(self.jpgs)
	def __getitem__(self, idx):
		jpg = self.jpgs[idx]
		#print(jpg)
		png = jpg.replace('jpg', 'png')
		im = cv2.imread(jpg)
		#im = cv2.resize(im, (1024, 512))
		im = im.transpose(2, 0, 1).astype(np.float32)
		im /= 255.0
		if not self.train: 
			#print(jpg)
			return torch.from_numpy(im)
		mask = cv2.imread(png, 0).astype(np.float32)
		mask /= 255.0
		#mask = cv2.resize(mask, (1024, 512))
		if np.random.random() < 0.5:
			im=im[:, :, ::-1].copy()
			mask = mask[:, ::-1].copy()
		return torch.from_numpy(im), torch.from_numpy(mask).long()
