# Data processing 
import pandas as pd
from skimage import io, transform
import cv2
# Math
import numpy as np
import math
# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
# Custom
from utils import vMF
'''
	input: (240, 320, 3) image
	output: two heads--> 1. distrubution (1D 240 vector), 2. parameters (1D 4 vector)
'''
class IlluminationModule(nn.Module):
	def __init__(self):
		super().__init__()
		self.cv_block1 = conv_bn_elu(3, 64, kernel_size=7, stride=2)
		self.cv_block2 = conv_bn_elu(64, 128, kernel_size=5, stride=2)
		self.cv_block3 = conv_bn_elu(128, 256, stride=2)
		self.cv_block4 = conv_bn_elu(256, 256)
		self.cv_block5 = conv_bn_elu(256, 256, stride=2)
		self.cv_block6= conv_bn_elu(256, 256)
		self.cv_block7 = conv_bn_elu(256, 256, stride=2)

		self.fc = nn.Linear(256*10*8, 2048)
		self.fc_bn = nn.BatchNorm1d(2048)
		''' two heads regression'''
		self.ds_fc = nn.Linear(2048, 256) # sky distribution
		self.ds_bn = nn.BatchNorm1d(256)
		self.pr_fc = nn.Linear(2048, 9) # sky and camera parameters
		self.pr_bn = nn.BatchNorm1d(9)
	
	def forward(self, x):
		#print(x.shape)
		x = self.cv_block1(x)
		#print(x.shape)
		x = self.cv_block2(x)
		x = self.cv_block3(x)
		x = self.cv_block4(x)
		x = self.cv_block5(x)
		x = self.cv_block6(x)
		x = self.cv_block7(x)
		#print(x.shape)
		x = x.view(-1, 256*10*8)
		x = F.elu(self.fc_bn(self.fc(x)))
		return F.log_softmax(self.ds_bn(self.ds_fc(x)), dim=1), self.pr_bn(self.pr_fc(x))

class IlluminationModule_Shadow(nn.Module):
	def __init__(self):
		super().__init__()
		self.cv_block1 = conv_bn_elu(4, 64, kernel_size=7, stride=2)
		self.cv_block2 = conv_bn_elu(64, 128, kernel_size=5, stride=2)
		self.cv_block3 = conv_bn_elu(128, 256, stride=2)
		self.cv_block4 = conv_bn_elu(256, 256)
		self.cv_block5 = conv_bn_elu(256, 256, stride=2)
		self.cv_block6= conv_bn_elu(256, 256)
		self.cv_block7 = conv_bn_elu(256, 256, stride=2)

		self.fc = nn.Linear(256*10*8, 2048)
		self.fc_bn = nn.BatchNorm1d(2048)
		''' two heads regression'''
		self.ds_fc = nn.Linear(2048, 256) # sky distribution
		self.ds_bn = nn.BatchNorm1d(256)
		self.pr_fc = nn.Linear(2048, 9) # sky and camera parameters
		self.pr_bn = nn.BatchNorm1d(9)
	
	def forward(self, x):
		#print(x.shape)
		x = self.cv_block1(x)
		#print(x.shape)
		x = self.cv_block2(x)
		x = self.cv_block3(x)
		x = self.cv_block4(x)
		x = self.cv_block5(x)
		x = self.cv_block6(x)
		x = self.cv_block7(x)
		#print(x.shape)
		x = x.view(-1, 256*10*8)
		x = F.elu(self.fc_bn(self.fc(x)))
		return F.log_softmax(self.ds_bn(self.ds_fc(x)), dim=1), self.pr_bn(self.pr_fc(x))


class IlluminationModule_SUN(nn.Module):
	def __init__(self):
		super().__init__()
		self.cv_block1 = conv_bn_elu(3, 64, kernel_size=7, stride=2) # 160 x 120
		self.cv_block2 = conv_bn_elu(64, 128, kernel_size=5, stride=2) # 80 x 60
		self.cv_block3 = conv_bn_elu(128, 256, stride=2) # 40 x 30
		self.cv_block4 = conv_bn_elu(256, 256)
		self.cv_block5 = conv_bn_elu(256, 256, stride=2)  # 20 x 15
		self.cv_block6= conv_bn_elu(256, 256)
		self.cv_block7 = conv_bn_elu(256, 256, stride=2) # 10 x 8

		self.fc = nn.Linear(256*10*8, 2048)
		self.fc_bn = nn.BatchNorm1d(2048)
		''' two heads regression'''
		self.ds_fc = nn.Linear(2048, 256) # sky distribution
		self.ds_bn = nn.BatchNorm1d(256)
		#self.pr_fc = nn.Linear(2048, 9) # sky and camera parameters
		#self.pr_bn = nn.BatchNorm1d(9)
	
	def forward(self, x):
		x = self.cv_block1(x)
		x = self.cv_block2(x)
		x = self.cv_block3(x)
		x = self.cv_block4(x)
		x = self.cv_block5(x)
		x = self.cv_block6(x)
		x = self.cv_block7(x)
		#print(x.shape)
		x = x.view(-1, 256*10*8)
		x = F.elu(self.fc_bn(self.fc(x)))
		return F.log_softmax(self.ds_bn(self.ds_fc(x)), dim=1)#, self.pr_bn(self.pr_fc(x))


class IlluminationModule_SUN_Shadow(nn.Module):
	def __init__(self):
		super().__init__()
		self.cv_block1 = conv_bn_elu(4, 64, kernel_size=7, stride=2) # 160 x 120
		self.cv_block2 = conv_bn_elu(64, 128, kernel_size=5, stride=2) # 80 x 60
		self.cv_block3 = conv_bn_elu(128, 256, stride=2) # 40 x 30
		self.cv_block4 = conv_bn_elu(256, 256)
		self.cv_block5 = conv_bn_elu(256, 256, stride=2)  # 20 x 15
		self.cv_block6= conv_bn_elu(256, 256)
		self.cv_block7 = conv_bn_elu(256, 256, stride=2) # 10 x 8

		self.fc = nn.Linear(256*10*8, 2048)
		self.fc_bn = nn.BatchNorm1d(2048)
		''' two heads regression'''
		self.ds_fc = nn.Linear(2048, 256) # sky distribution
		self.ds_bn = nn.BatchNorm1d(256)
		#self.pr_fc = nn.Linear(2048, 9) # sky and camera parameters
		#self.pr_bn = nn.BatchNorm1d(9)
	
	def forward(self, x):
		x = self.cv_block1(x)
		x = self.cv_block2(x)
		x = self.cv_block3(x)
		x = self.cv_block4(x)
		x = self.cv_block5(x)
		x = self.cv_block6(x)
		x = self.cv_block7(x)
		#print(x.shape)
		x = x.view(-1, 256*10*8)
		x = F.elu(self.fc_bn(self.fc(x)))
		return F.log_softmax(self.ds_bn(self.ds_fc(x)), dim=1)#, self.pr_bn(self.pr_fc(x))
		

class AlexNetModule(nn.Module):
	def __init__(self):
		super().__init__()
		self.cv_block1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
		self.cv_block2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
		self.cv_block3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
		self.cv_block4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
		self.cv_block5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.mp = nn.MaxPool2d(kernel_size=3, stride=2)
		self.elu = nn.ELU()
		self.ln1 = nn.Linear(13824, 4096)
		self.ln2 = nn.Linear(4096, 4096)
		self.fc = nn.Linear(4096, 2048)
		self.fc_bn = nn.BatchNorm1d(2048)

		''' two heads regression'''
		self.ds_fc = nn.Linear(2048, 256) # sky distribution
		self.ds_bn = nn.BatchNorm1d(256)
		self.pr_fc = nn.Linear(2048, 9) # sky and camera parameters
		self.pr_bn = nn.BatchNorm1d(9)
		self.dropout = nn.Dropout()
        
	def forward(self, x):
		x = self.cv_block1(x)
		x = self.elu(x)
		x = self.mp(x)

		x = self.cv_block2(x)
		x = self.elu(x)
		x = self.mp(x)

		x = self.cv_block3(x)
		x = self.elu(x)

		x = self.cv_block4(x)
		x = self.elu(x)
        
		x = self.cv_block5(x)
		x = self.elu(x)
		x = self.mp(x)

		x = x.view(x.size(0), -1)

		x = self.dropout(x)
		x = self.ln1(x)
		x = self.elu(x)
		x = self.dropout(x)
		x = self.ln2(x)
		x = self.elu(x)
		x = F.elu(self.fc_bn(self.fc(x)))
		return F.log_softmax(self.ds_bn(self.ds_fc(x)), dim=1), self.pr_bn(self.pr_fc(x))

class AlexNetModule_Shadow(nn.Module):
	def __init__(self):
		super().__init__()
		self.cv_block1 = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
		self.cv_block2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
		self.cv_block3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
		self.cv_block4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
		self.cv_block5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.mp = nn.MaxPool2d(kernel_size=3, stride=2)
		self.elu = nn.ELU()
		self.ln1 = nn.Linear(13824, 4096)
		self.ln2 = nn.Linear(4096, 4096)
		self.fc = nn.Linear(4096, 2048)
		self.fc_bn = nn.BatchNorm1d(2048)

		''' two heads regression'''
		self.ds_fc = nn.Linear(2048, 256) # sky distribution
		self.ds_bn = nn.BatchNorm1d(256)
		self.pr_fc = nn.Linear(2048, 9) # sky and camera parameters
		self.pr_bn = nn.BatchNorm1d(9)
		self.dropout = nn.Dropout()
        
	def forward(self, x):
		x = self.cv_block1(x)
		x = self.elu(x)
		x = self.mp(x)

		x = self.cv_block2(x)
		x = self.elu(x)
		x = self.mp(x)

		x = self.cv_block3(x)
		x = self.elu(x)

		x = self.cv_block4(x)
		x = self.elu(x)
        
		x = self.cv_block5(x)
		x = self.elu(x)
		x = self.mp(x)

		x = x.view(x.size(0), -1)

		x = self.dropout(x)
		x = self.ln1(x)
		x = self.elu(x)
		x = self.dropout(x)
		x = self.ln2(x)
		x = self.elu(x)
		x = F.elu(self.fc_bn(self.fc(x)))
		return F.log_softmax(self.ds_bn(self.ds_fc(x)), dim=1), self.pr_bn(self.pr_fc(x))

class AlexNetModule_SUN(nn.Module):
	def __init__(self):
		super().__init__()
		self.cv_block1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
		self.cv_block2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
		self.cv_block3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
		self.cv_block4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
		self.cv_block5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.mp = nn.MaxPool2d(kernel_size=3, stride=2)
		self.elu = nn.ELU()
		self.ln1 = nn.Linear(13824, 4096)
		self.ln2 = nn.Linear(4096, 4096)
		self.fc = nn.Linear(4096, 2048)
		self.fc_bn = nn.BatchNorm1d(2048)

		''' two heads regression'''
		self.ds_fc = nn.Linear(2048, 256) # sky distribution
		self.ds_bn = nn.BatchNorm1d(256)
		self.dropout = nn.Dropout()
        
	def forward(self, x):
		x = self.cv_block1(x)
		x = self.elu(x)
		x = self.mp(x)

		x = self.cv_block2(x)
		x = self.elu(x)
		x = self.mp(x)

		x = self.cv_block3(x)
		x = self.elu(x)

		x = self.cv_block4(x)
		x = self.elu(x)
        
		x = self.cv_block5(x)
		x = self.elu(x)
		x = self.mp(x)

		x = x.view(x.size(0), -1)

		x = self.dropout(x)
		x = self.ln1(x)
		x = self.elu(x)
		x = self.dropout(x)
		x = self.ln2(x)
		x = self.elu(x)
		x = F.elu(self.fc_bn(self.fc(x)))
		return F.log_softmax(self.ds_bn(self.ds_fc(x)), dim=1)

class AlexNetModule_SUN_Shadow(nn.Module):
	def __init__(self):
		super().__init__()
		self.cv_block1 = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
		self.cv_block2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
		self.cv_block3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
		self.cv_block4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
		self.cv_block5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.mp = nn.MaxPool2d(kernel_size=3, stride=2)
		self.elu = nn.ELU()
		self.ln1 = nn.Linear(13824, 4096)
		self.ln2 = nn.Linear(4096, 4096)
		self.fc = nn.Linear(4096, 2048)
		self.fc_bn = nn.BatchNorm1d(2048)

		''' two heads regression'''
		self.ds_fc = nn.Linear(2048, 256) # sky distribution
		self.ds_bn = nn.BatchNorm1d(256)
		self.dropout = nn.Dropout()
        
	def forward(self, x):
		x = self.cv_block1(x)
		x = self.elu(x)
		x = self.mp(x)

		x = self.cv_block2(x)
		x = self.elu(x)
		x = self.mp(x)

		x = self.cv_block3(x)
		x = self.elu(x)

		x = self.cv_block4(x)
		x = self.elu(x)
        
		x = self.cv_block5(x)
		x = self.elu(x)
		x = self.mp(x)

		x = x.view(x.size(0), -1)

		x = self.dropout(x)
		x = self.ln1(x)
		x = self.elu(x)
		x = self.dropout(x)
		x = self.ln2(x)
		x = self.elu(x)
		x = F.elu(self.fc_bn(self.fc(x)))
		return F.log_softmax(self.ds_bn(self.ds_fc(x)), dim=1)


def conv_bn_elu(in_, out_, kernel_size=3, stride=1, padding=True):
	## conv layer with BN and ELU function 
	pad = int(kernel_size/2)
	if padding is False:
		pad = 0
	return nn.Sequential(
		nn.Conv2d(in_, out_, kernel_size, stride=stride, padding=pad),
		nn.BatchNorm2d(out_),
		nn.ELU(),
	)

'''
	Dataset loader 
	dataset standardization ==> 
		mean: [0.48548178 0.48455666 0.46329196] std: [0.21904471 0.21578524 0.23359051]
'''
class Train_Dataset_SUN(Dataset):
	def __init__(self, csv_path):
		self.to_tensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196],
												std= [0.21904471, 0.21578524, 0.23359051])
		self.data = pd.read_csv(csv_path)
		self.data = self.data[self.data['mode']<=0.7]
		# df[df['mode']>=0.7]
		self.path_arr = np.asarray(self.data['filepath'])

		self.theta_arr = np.asarray(self.data['u'])
		self.phi_arr = np.asarray(self.data['v'])
		
		self.wsky1 = np.asarray(self.data['wsky1'])
		self.wsky2 = np.asarray(self.data['wsky2'])
		self.wsky3 = np.asarray(self.data['wsky3'])
		self.wsun1 = np.asarray(self.data['wsun1'])
		self.wsun2 = np.asarray(self.data['wsun2'])
		self.wsun3 = np.asarray(self.data['wsun3'])
		self.kappa = np.asarray(self.data['kappa'])
		self.beta = np.asarray(self.data['beta'])
		self.turbidity = np.asarray(self.data['turbidity'])
		# ,file_name, azimuth, elevation, turbidity, exposure, fov, pitch

		self.data_len = len(self.data.index)

	def __getitem__(self, index):
		source_img_name = self.path_arr[index]
		source_img = io.imread(source_img_name)[:, :, :3]/255.0
		tensor_img = self.to_tensor(source_img)
		sun_pos = np.asarray([self.theta_arr[index], self.phi_arr[index]])
		sp_pdf = vMF(sun_pos) # target probability distribution of sun position
		#pr_vec = np.asarray([self.wsky1[index], self.wsky2[index], self.wsky3[index], self.wsun1[index], self.wsun2[index], self.wsun3[index],
		#					 self.kappa[index], self.beta[index], self.turbidity[index]])#, math.radians(float(self.fov[index]))]) # target parameters
		label = {'img': self.normalize(tensor_img), 'dis': sp_pdf}#, 'prrs': pr_vec}
		return label

	def __len__(self):
		return self.data_len

class Train_Dataset_SUN_Shadow(Dataset):
	def __init__(self, csv_path):
		self.to_tensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196, 0.5],
												std= [0.21904471, 0.21578524, 0.23359051, 0.5])
		self.data = pd.read_csv(csv_path)
		self.data = self.data[self.data['mode']<=0.7]
		# df[df['mode']>=0.7]
		self.path_arr = np.asarray(self.data['filepath'])
		self.concat_arr = np.asarray(self.data['shadow_path'])

		self.theta_arr = np.asarray(self.data['u'])
		self.phi_arr = np.asarray(self.data['v'])
		
		self.wsky1 = np.asarray(self.data['wsky1'])
		self.wsky2 = np.asarray(self.data['wsky2'])
		self.wsky3 = np.asarray(self.data['wsky3'])
		self.wsun1 = np.asarray(self.data['wsun1'])
		self.wsun2 = np.asarray(self.data['wsun2'])
		self.wsun3 = np.asarray(self.data['wsun3'])
		self.kappa = np.asarray(self.data['kappa'])
		self.beta = np.asarray(self.data['beta'])
		self.turbidity = np.asarray(self.data['turbidity'])
		# ,file_name, azimuth, elevation, turbidity, exposure, fov, pitch

		self.data_len = len(self.data.index)

	def __getitem__(self, index):
		source_img_name = self.path_arr[index]
		source_img = cv2.imread(source_img_name)[:, :, :3]/255.0
		concat_img_name = self.concat_arr[index]
		concat_img = cv2.imread(concat_img_name)
		concat_img = cv2.cvtColor(concat_img, cv2.COLOR_BGR2GRAY)[:]/255
		merge_img = cv2.merge([source_img,concat_img])
		tensor_img = self.to_tensor(merge_img)
		sun_pos = np.asarray([self.theta_arr[index], self.phi_arr[index]])
		sp_pdf = vMF(sun_pos) # target probability distribution of sun position
		#pr_vec = np.asarray([self.wsky1[index], self.wsky2[index], self.wsky3[index], self.wsun1[index], self.wsun2[index], self.wsun3[index],
		#					 self.kappa[index], self.beta[index], self.turbidity[index]])#, math.radians(float(self.fov[index]))]) # target parameters
		label = {'img': self.normalize(tensor_img), 'dis': sp_pdf}#, 'prrs': pr_vec}
		return label

	def __len__(self):
		return self.data_len

class Eval_Dataset_SUN(Dataset):
	def __init__(self, csv_path):
		self.to_tensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196],
												std= [0.21904471, 0.21578524, 0.23359051])
		self.data = pd.read_csv(csv_path)
		self.data = self.data[self.data['mode']>0.7]
		self.path_arr = np.asarray(self.data['filepath'])

		self.theta_arr = np.asarray(self.data['u'])
		self.phi_arr = np.asarray(self.data['v'])
		
		self.wsky1 = np.asarray(self.data['wsky1'])
		self.wsky2 = np.asarray(self.data['wsky2'])
		self.wsky3 = np.asarray(self.data['wsky3'])
		self.wsun1 = np.asarray(self.data['wsun1'])
		self.wsun2 = np.asarray(self.data['wsun2'])
		self.wsun3 = np.asarray(self.data['wsun3'])
		self.kappa = np.asarray(self.data['kappa'])
		self.beta = np.asarray(self.data['beta'])
		self.turbidity = np.asarray(self.data['turbidity'])
		self.fov = np.asarray(self.data['fov'])

		self.data_len = len(self.data.index)

	def __getitem__(self, index):
		source_img_name = self.path_arr[index]
		source_img = io.imread(source_img_name)[:, :, :3]/255.0
		tensor_img = self.to_tensor(source_img)
		sun_pos = np.asarray([self.theta_arr[index], self.phi_arr[index]])
		sp_pdf = vMF(sun_pos) # target probability distribution of sun position
		#pr_vec = np.asarray([self.wsky1[index], self.wsky2[index], self.wsky3[index], self.wsun1[index], self.wsun2[index], self.wsun3[index],
		#					 self.kappa[index], self.beta[index], self.turbidity[index]])#, math.radians(float(self.fov[index]))]) # target parameters
		label = {'img': self.normalize(tensor_img), 'dis': sp_pdf, 'sp': sun_pos}#, 'prrs': pr_vec, 
		return label

	def __len__(self):
		return self.data_len

class Eval_Dataset_SUN_Shadow(Dataset):
	def __init__(self, csv_path):
		self.to_tensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196, 0.5],
												std= [0.21904471, 0.21578524, 0.23359051, 0.5])
		self.data = pd.read_csv(csv_path)
		self.data = self.data[self.data['mode']>0.7]
		self.path_arr = np.asarray(self.data['filepath'])
		self.concat_arr = np.asarray(self.data['shadow_path'])

		self.theta_arr = np.asarray(self.data['u'])
		self.phi_arr = np.asarray(self.data['v'])
		
		self.wsky1 = np.asarray(self.data['wsky1'])
		self.wsky2 = np.asarray(self.data['wsky2'])
		self.wsky3 = np.asarray(self.data['wsky3'])
		self.wsun1 = np.asarray(self.data['wsun1'])
		self.wsun2 = np.asarray(self.data['wsun2'])
		self.wsun3 = np.asarray(self.data['wsun3'])
		self.kappa = np.asarray(self.data['kappa'])
		self.beta = np.asarray(self.data['beta'])
		self.turbidity = np.asarray(self.data['turbidity'])
		self.fov = np.asarray(self.data['fov'])

		self.data_len = len(self.data.index)

	def __getitem__(self, index):
		source_img_name = self.path_arr[index]
		source_img = cv2.imread(source_img_name)[:, :, :3]/255.0
		concat_img_name = self.concat_arr[index]
		concat_img = cv2.imread(concat_img_name)
		concat_img = cv2.cvtColor(concat_img, cv2.COLOR_BGR2GRAY)[:]/255
		merge_img = cv2.merge([source_img,concat_img])
		tensor_img = self.to_tensor(merge_img)
		sun_pos = np.asarray([self.theta_arr[index], self.phi_arr[index]])
		sp_pdf = vMF(sun_pos) # target probability distribution of sun position
		#pr_vec = np.asarray([self.wsky1[index], self.wsky2[index], self.wsky3[index], self.wsun1[index], self.wsun2[index], self.wsun3[index],
		#					 self.kappa[index], self.beta[index], self.turbidity[index]])#, math.radians(float(self.fov[index]))]) # target parameters
		label = {'img': self.normalize(tensor_img), 'dis': sp_pdf, 'sp': sun_pos}#, 'prrs': pr_vec, 
		return label

	def __len__(self):
		return self.data_len

class Inference_Data(Dataset):
	def __init__(self, img_path):
		self.input_img = io.imread(img_path)
		self.to_tensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196],
												std= [0.21904471, 0.21578524, 0.23359051])
		self.data_len = 1

	def __getitem__(self, index):
		tensor_img = self.to_tensor(self.input_img)
		return self.normalize(tensor_img)

	def __len__(self):
		return self.data_len

class Train_Dataset(Dataset):
	def __init__(self, csv_path):
		self.to_tensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196],
												std= [0.21904471, 0.21578524, 0.23359051])
		self.data = pd.read_csv(csv_path)
		self.data = self.data[self.data['mode']<=0.7]
		# df[df['mode']>=0.7]
		self.path_arr = np.asarray(self.data['filepath'])

		self.theta_arr = np.asarray(self.data['u'])
		self.phi_arr = np.asarray(self.data['v'])
		
		self.wsky1 = np.asarray(self.data['wsky1'])
		self.wsky2 = np.asarray(self.data['wsky2'])
		self.wsky3 = np.asarray(self.data['wsky3'])
		self.wsun1 = np.asarray(self.data['wsun1'])
		self.wsun2 = np.asarray(self.data['wsun2'])
		self.wsun3 = np.asarray(self.data['wsun3'])
		self.kappa = np.asarray(self.data['kappa'])
		self.beta = np.asarray(self.data['beta'])
		self.turbidity = np.asarray(self.data['turbidity'])
		# ,file_name, azimuth, elevation, turbidity, exposure, fov, pitch

		self.data_len = len(self.data.index)

	def __getitem__(self, index):
		source_img_name = self.path_arr[index]
		source_img = io.imread(source_img_name)[:, :, :3]/255.0
		tensor_img = self.to_tensor(source_img)
		sun_pos = np.asarray([self.theta_arr[index], self.phi_arr[index]])
		sp_pdf = vMF(sun_pos) # target probability distribution of sun position
		pr_vec = np.asarray([self.wsky1[index], self.wsky2[index], self.wsky3[index], self.wsun1[index], self.wsun2[index], self.wsun3[index],
							 self.kappa[index], self.beta[index], self.turbidity[index]])#, math.radians(float(self.fov[index]))]) # target parameters
		label = {'img': self.normalize(tensor_img), 'dis': sp_pdf, 'prrs': pr_vec}
		return label

	def __len__(self):
		return self.data_len

class Eval_Dataset(Dataset):
	def __init__(self, csv_path):
		self.to_tensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196],
												std= [0.21904471, 0.21578524, 0.23359051])
		self.data = pd.read_csv(csv_path)
		self.data = self.data[self.data['mode']>0.7]
		self.path_arr = np.asarray(self.data['filepath'])

		self.theta_arr = np.asarray(self.data['u'])
		self.phi_arr = np.asarray(self.data['v'])
		
		self.wsky1 = np.asarray(self.data['wsky1'])
		self.wsky2 = np.asarray(self.data['wsky2'])
		self.wsky3 = np.asarray(self.data['wsky3'])
		self.wsun1 = np.asarray(self.data['wsun1'])
		self.wsun2 = np.asarray(self.data['wsun2'])
		self.wsun3 = np.asarray(self.data['wsun3'])
		self.kappa = np.asarray(self.data['kappa'])
		self.beta = np.asarray(self.data['beta'])
		self.turbidity = np.asarray(self.data['turbidity'])
		self.fov = np.asarray(self.data['fov'])

		self.data_len = len(self.data.index)

	def __getitem__(self, index):
		source_img_name = self.path_arr[index]
		source_img = io.imread(source_img_name)[:, :, :3]/255.0
		tensor_img = self.to_tensor(source_img)
		sun_pos = np.asarray([self.theta_arr[index], self.phi_arr[index]])
		sp_pdf = vMF(sun_pos) # target probability distribution of sun position
		pr_vec = np.asarray([self.wsky1[index], self.wsky2[index], self.wsky3[index], self.wsun1[index], self.wsun2[index], self.wsun3[index],
							 self.kappa[index], self.beta[index], self.turbidity[index]])#, math.radians(float(self.fov[index]))]) # target parameters
		label = {'img': self.normalize(tensor_img), 'dis': sp_pdf, 'prrs': pr_vec, 'sp': sun_pos}
		return label

	def __len__(self):
		return self.data_len

class Train_Shadow_Dataset(Dataset):
	def __init__(self, csv_path):
		self.to_tensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196, 0.5],
												std= [0.21904471, 0.21578524, 0.23359051, 0.5])
		self.data = pd.read_csv(csv_path)
		self.data = self.data[self.data['mode']<=0.7]
		# df[df['mode']>=0.7]
		self.path_arr = np.asarray(self.data['filepath'])
		self.concat_arr = np.asarray(self.data['shadow_path'])

		self.theta_arr = np.asarray(self.data['u'])
		self.phi_arr = np.asarray(self.data['v'])
		
		self.wsky1 = np.asarray(self.data['wsky1'])
		self.wsky2 = np.asarray(self.data['wsky2'])
		self.wsky3 = np.asarray(self.data['wsky3'])
		self.wsun1 = np.asarray(self.data['wsun1'])
		self.wsun2 = np.asarray(self.data['wsun2'])
		self.wsun3 = np.asarray(self.data['wsun3'])
		self.kappa = np.asarray(self.data['kappa'])
		self.beta = np.asarray(self.data['beta'])
		self.turbidity = np.asarray(self.data['turbidity'])
		# ,file_name, azimuth, elevation, turbidity, exposure, fov, pitch

		self.data_len = len(self.data.index)

	def __getitem__(self, index):
		source_img_name = self.path_arr[index]
		source_img = cv2.imread(source_img_name)[:, :, :3]/255.0
		concat_img_name = self.concat_arr[index]
		concat_img = cv2.imread(concat_img_name)
		concat_img = cv2.cvtColor(concat_img, cv2.COLOR_BGR2GRAY)[:]/255
		merge_img = cv2.merge([source_img,concat_img])
		tensor_img = self.to_tensor(merge_img)

		sun_pos = np.asarray([self.theta_arr[index], self.phi_arr[index]])
		sp_pdf = vMF(sun_pos) # target probability distribution of sun position
		pr_vec = np.asarray([self.wsky1[index], self.wsky2[index], self.wsky3[index], self.wsun1[index], self.wsun2[index], self.wsun3[index],
							 self.kappa[index], self.beta[index], self.turbidity[index]])#, math.radians(float(self.fov[index]))]) # target parameters
		label = {'img': self.normalize(tensor_img), 'dis': sp_pdf, 'prrs': pr_vec}
		#print("tensor_img", tensor_img)
		return label

	def __len__(self):
		return self.data_len

class Eval_Shadow_Dataset(Dataset):
	def __init__(self, csv_path):
		self.to_tensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196, 0.5],
												std= [0.21904471, 0.21578524, 0.23359051, 0.5])
		self.data = pd.read_csv(csv_path)
		self.data = self.data[self.data['mode']>0.7]
		self.path_arr = np.asarray(self.data['filepath'])
		self.concat_arr = np.asarray(self.data['shadow_path'])

		self.theta_arr = np.asarray(self.data['u'])
		self.phi_arr = np.asarray(self.data['v'])
		
		self.wsky1 = np.asarray(self.data['wsky1'])
		self.wsky2 = np.asarray(self.data['wsky2'])
		self.wsky3 = np.asarray(self.data['wsky3'])
		self.wsun1 = np.asarray(self.data['wsun1'])
		self.wsun2 = np.asarray(self.data['wsun2'])
		self.wsun3 = np.asarray(self.data['wsun3'])
		self.kappa = np.asarray(self.data['kappa'])
		self.beta = np.asarray(self.data['beta'])
		self.turbidity = np.asarray(self.data['turbidity'])
		self.fov = np.asarray(self.data['fov'])

		self.data_len = len(self.data.index)

	def __getitem__(self, index):
		source_img_name = self.path_arr[index]
		source_img = cv2.imread(source_img_name)[:, :, :3]/255.0
		concat_img_name = self.concat_arr[index]
		concat_img = cv2.imread(concat_img_name)
		concat_img = cv2.cvtColor(concat_img, cv2.COLOR_BGR2GRAY)[:]/255
		merge_img = cv2.merge([source_img,concat_img])
		tensor_img = self.to_tensor(merge_img)

		sun_pos = np.asarray([self.theta_arr[index], self.phi_arr[index]])
		sp_pdf = vMF(sun_pos) # target probability distribution of sun position
		pr_vec = np.asarray([self.wsky1[index], self.wsky2[index], self.wsky3[index], self.wsun1[index], self.wsun2[index], self.wsun3[index],
							 self.kappa[index], self.beta[index], self.turbidity[index]])#, math.radians(float(self.fov[index]))]) # target parameters
		label = {'img': self.normalize(tensor_img), 'dis': sp_pdf, 'prrs': pr_vec, 'sp': sun_pos}
		return label

	def __len__(self):
		return self.data_len