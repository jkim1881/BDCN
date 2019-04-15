import os
import cv2
import numpy as np
from PIL import Image
import scipy.io
import torch
from torch.utils import data
import random
from cStringIO import StringIO


def load_image_with_cache(path, cache=None, lock=None, matfile=False):
	if cache is not None:
		if not cache.has_key(path):
			with open(path, 'rb') as f:
				cache[path] = f.read()
		if matfile:
			return scipy.io.loadmat(StringIO(cache[path]))
		else:
			return Image.open(StringIO(cache[path]))
	else:
		if matfile:
			return scipy.io.loadmat(path)
		else:
			return Image.open(path)


class Data(data.Dataset):
	def __init__(self, root, lst, yita=0.5,
		mean_bgr = np.array([104.00699, 116.66877, 122.67892]),
		crop_size=None, rgb=True, scale=None):
		self.mean_bgr = mean_bgr
		self.root = root
		self.lst = lst
		self.yita = yita
		self.crop_size = crop_size
		self.rgb = rgb
		self.scale = scale
		self.cache = {}

		lst_dir = os.path.join(self.root, self.lst)
		# self.files = np.loadtxt(lst_dir, dtype=str)
		with open(lst_dir, 'r') as f:
			self.files = f.readlines()
			self.files = [line.strip().split(' ') for line in self.files]

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		data_file = self.files[index]
		# load Image
		img_file = self.root + data_file[0]
		# print(img_file)
		if not os.path.exists(img_file):
			img_file = img_file.replace('jpg', 'png')
		# img = Image.open(img_file)
		img = load_image_with_cache(img_file, self.cache)
		# load gt image
		gt_file = self.root + data_file[1]
		# gt = Image.open(gt_file)
		gt = load_image_with_cache(gt_file, self.cache)
		if gt.mode == '1':
			gt  = gt.convert('L')
		return self.transform(img, gt)

	def transform(self, img, gt):
		gt = np.array(gt, dtype=np.float32)
		if len(gt.shape) == 3:
			gt = gt[:, :, 0]
		gt /= 255.
		gt[gt >= self.yita] = 1
		gt = torch.from_numpy(np.array([gt])).float()
		img = np.array(img, dtype=np.float32)
		if self.rgb:
			img = img[:, :, ::-1] # RGB->BGR
		img -= self.mean_bgr
		data = []
		if self.scale is not None:
			for scl in self.scale:
				img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
				data.append(torch.from_numpy(img_scale.transpose((2,0,1))).float())
			return data, gt
		img = img.transpose((2, 0, 1))
		img = torch.from_numpy(img.copy()).float()
		if self.crop_size:
			_, h, w = gt.size()
			assert(self.crop_size < h and self.crop_size < w)
			i = random.randint(0, h - self.crop_size)
			j = random.randint(0, w - self.crop_size)
			img = img[:, i:i+self.crop_size, j:j+self.crop_size]
			gt = gt[:, i:i+self.crop_size, j:j+self.crop_size]
		return img, gt


class BSDS_data_jk(data.Dataset):
	def __init__(self, root, type, yita=0.5,
		mean_bgr = np.array([104.00699, 116.66877, 122.67892]),
		crop_size=None, rgb=True, scale=None):
		self.mean_bgr = mean_bgr
		self.root = root
		self.type = type
		self.yita = yita
		self.crop_size = crop_size
		self.rgb = rgb
		self.scale = scale
		self.cache = {}

		# get list of images and gts from a specified path
		self.img_ext = '.jpg'
		self.gt_ext = '.mat'
		image_dir = os.path.join(self.root, 'data', 'images', self.type)
		gt_dir = os.path.join(self.root, 'data', 'gt', self.type)
		image_list = os.listdir(image_dir)
		gt_list = os.listdir(gt_dir)
		image_filenames = [file.split('.')[0] for file in image_list if self.img_ext in file]
		gt_filenames = [file.split('.')[0] for file in gt_list if self.gt_ext in file]

		# sanity check
		if not(image_filenames == gt_filenames):
			raise ValueError('image_filenames and gt_filenames do not match.')
		else:
			self.files = image_filenames

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		# load Image
		img_file = os.path.join(self.root, 'data', 'images', self.files[index] + self.img_ext)
		if not os.path.exists(img_file):
			raise ValueError('Cannot find image by path :' + img_file)
		img = load_image_with_cache(img_file, self.cache)
		# load gt image
		gt_file = os.path.join(self.root, 'data', 'gt', self.files[index] + self.gt_ext)
		gt = load_image_with_cache(gt_file, self.cache, matfile=True)
		return self.transform(img, gt)

	def transform(self, img, gt):
		gt = np.array(gt, dtype=np.float32)
		if len(gt.shape) == 3:
			gt = gt[:, :, 0]
		gt /= 255.
		gt[gt >= self.yita] = 1
		gt = torch.from_numpy(np.array([gt])).float()
		img = np.array(img, dtype=np.float32)
		if self.rgb:
			img = img[:, :, ::-1] # RGB->BGR
		img -= self.mean_bgr
		data = []
		if self.scale is not None:
			for scl in self.scale:
				img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
				data.append(torch.from_numpy(img_scale.transpose((2,0,1))).float())
			return data, gt
		img = img.transpose((2, 0, 1))
		img = torch.from_numpy(img.copy()).float()
		if self.crop_size:
			_, h, w = gt.size()
			assert(self.crop_size < h and self.crop_size < w)
			i = random.randint(0, h - self.crop_size)
			j = random.randint(0, w - self.crop_size)
			img = img[:, i:i+self.crop_size, j:j+self.crop_size]
			gt = gt[:, i:i+self.crop_size, j:j+self.crop_size]
		return img, gt
