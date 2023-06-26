import os, sys, cv2
import numpy as np
import torch, glob
from bisenetv2 import BiSeNetV2
from torch.utils.data import DataLoader
from data import LWDDataset

bs = 1
# data
val_ds = LWDDataset("/home/lwd/data/test", False)
n_val = len(val_ds)/bs
val_loader = DataLoader(val_ds, shuffle=False, batch_size=bs, pin_memory=True, drop_last = False)
# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=BiSeNetV2(2)
model_path='result/model/244--0.07386.pth'
paras=torch.load(model_path, map_location='cuda')
model.load_state_dict(paras)
model.to(device=device)
model.eval()

palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
for it, im in enumerate(val_loader):
	im = im.cuda()
	logits, *logits_aux = model(im)
	#print(logits.shape)
	res = logits.argmax(dim=1)
	res = res.squeeze().cpu().numpy().astype('uint8')
	#res=palette[res]
	res[res>0] = 255
	im=im[0]*255.0
	im = im.permute(1, 2, 0).cpu().numpy().astype('uint8')
	tmp = np.zeros(im.shape).astype('uint8')
	tmp[..., 2] = res/2
	im[res>0] = im[res>0] / 2 + tmp[res>0]
	cv2.imshow('ss', im)
	if cv2.waitKey() & 0xff == 27: break
