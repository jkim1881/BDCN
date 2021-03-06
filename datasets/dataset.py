import os
import cv2
import numpy as np
from PIL import Image
import scipy.io
import scipy.misc
import torch
from torch.utils import data
import random
from cStringIO import StringIO

def load_image_with_cache(path, cache=None, lock=None):
	if cache is not None:
		if not cache.has_key(path):
			with open(path, 'rb') as f:
				cache[path] = f.read()
		return Image.open(StringIO(cache[path]))
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

class Data_tv(data.Dataset):
	def __init__(self, root, lst, yita=0.5,
		mean_bgr = np.array([104.00699, 116.66877, 122.67892]),
		crop_size=None, rgb=True, scale=None,
		random_sample=False, front_or_end='front', use_ratio=1.0):
		self.mean_bgr = mean_bgr
		self.root = root
		self.lst = lst
		self.yita = yita
		self.crop_size = crop_size
		self.rgb = rgb
		self.scale = scale
		self.cache = {}

		self.random_sample = random_sample
		self.use_ratio = use_ratio
		self.front_or_end = front_or_end

		lst_dir = os.path.join(self.root, self.lst)
		# self.files = np.loadtxt(lst_dir, dtype=str)
		with open(lst_dir, 'r') as f:
			self.files = f.readlines()
			self.files = [line.strip().split(' ') for line in self.files]
		if use_ratio < 1.0:
			if self.random_sample:
				import random
				random.shuffle(self.files)
			if front_or_end == 'front':
				self.files = self.files[:int(self.use_ratio * len(self.files))]
			elif front_or_end == 'end':
				self.files = self.files[int(self.use_ratio * len(self.files)):]

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


def load_image_with_cache_bsds_crops(path, cache=None, lock=None, npy=False):
	if cache is not None:
		if not cache.has_key(path):
			with open(path, 'rb') as f:
				cache[path] = f.read()
		if npy:
			return np.load(path)
		else:
			return scipy.misc.imread(StringIO(cache[path]))
			# return Image.open(StringIO(cache[path]))
	else:
		if npy:
			return np.load(path)
		else:
			return scipy.misc.imread(path)
			# return Image.open(path)

class BSDS_crops(data.Dataset):
	def __init__(self, root, type, yita=0.5,
		mean_bgr = np.array([104.00699, 116.66877, 122.67892]),
		crop_size=None, rgb=True, scale=None,
		max_examples=None, random_sample=False, return_filename=False):
		self.mean_bgr = mean_bgr
		self.root = root
		self.type = type
		self.yita = yita
		self.crop_size = crop_size
		self.rgb = rgb
		self.max_examples = max_examples
		self.random_sample = random_sample
		self.return_filename=return_filename
		self.scale = scale
		self.cache = {}

		# get list of images and gts from a specified path
		self.img_ext = '.jpg'
		self.gt_ext = '.npy'
		image_dir = os.path.join(self.root, 'data', 'images', self.type)
		gt_dir = os.path.join(self.root, 'data', 'groundTruth', self.type)
		image_list = os.listdir(image_dir)
		gt_list = os.listdir(gt_dir)
		image_filenames_int = [file.split('.')[0] for file in image_list if self.img_ext in file]
		gt_filenames_int = [file.split('.')[0] for file in gt_list if self.gt_ext in file]
		image_filenames_int.sort()
		gt_filenames_int.sort()

		# sanity check
		if not(image_filenames_int == gt_filenames_int):
			raise ValueError('image_filenames and gt_filenames do not match.')
		else:
			if self.max_examples is not None:
				if self.random_sample:
					import random
					random.shuffle(image_filenames_int)
				image_filenames_int = image_filenames_int[:self.max_examples]
			self.files = [str(integer) for integer in image_filenames_int]

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		# load Image
		img_file = os.path.join(self.root, 'data', 'images', self.type, self.files[index] + self.img_ext)
		if not os.path.exists(img_file):
			raise ValueError('Cannot find image by path :' + img_file)
		img = load_image_with_cache_bsds_crops(img_file, cache=None) #self.cache)
		# load gt image
		gt_file = os.path.join(self.root, 'data', 'groundTruth', self.type, self.files[index] + self.gt_ext)
		gt = load_image_with_cache_bsds_crops(gt_file, cache=None, npy=True) #self.cache, matfile=True)
		if not self.return_filename:
			return self.transform(img, gt)
		else:
			return self.transform(img, gt), self.files[index] + self.img_ext

	def transform(self, img, gt):
		if len(gt.shape) == 3:
			gt = gt[:, :, 0]
		# gt_mean /= 255.
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
			_, h, w = gt.shape
			assert(self.crop_size < h and self.crop_size < w)
			i = random.randint(0, h - self.crop_size)
			j = random.randint(0, w - self.crop_size)
			img = img[:, i:i+self.crop_size, j:j+self.crop_size]
			gt = gt[:, i:i+self.crop_size, j:j+self.crop_size]
		return img, gt

def load_image_with_cache_multicue_crops(path, cache=None, lock=None, npy=False):
	if cache is not None:
		if not cache.has_key(path):
			with open(path, 'rb') as f:
				cache[path] = f.read()
		if npy:
			return np.load(path)
		else:
			return scipy.misc.imread(StringIO(cache[path]))
			# return Image.open(StringIO(cache[path]))
	else:
		if npy:
			return np.load(path)
		else:
			return scipy.misc.imread(path)
			# return Image.open(path)

class Multicue_crops(data.Dataset):
	def __init__(self, root, type, task, yita=0.5,
		mean_bgr = np.array([104.00699, 116.66877, 122.67892]),
		crop_size=None, rgb=True, scale=None,
		max_examples=None, random_sample=False, return_filename=False):
		self.mean_bgr = mean_bgr
		self.root = root
		self.type = type # train or test
		self.task = task # edges or boundaries
		self.yita = yita
		self.crop_size = crop_size
		self.rgb = rgb
		self.max_examples = max_examples
		self.random_sample = random_sample
		self.return_filename = return_filename
		self.scale = scale
		self.cache = {}
		if not (self.task=='edges' or self.task=='boundaries'):
			return ValueError('task should either be edges or boundaries.')

		# get list of images and gts from a specified path
		self.img_ext = '.jpg'
		self.gt_ext = '.npy'
		image_dir = os.path.join(self.root, 'data', 'images', self.type)
		gt_dir = os.path.join(self.root, 'data', 'groundTruth', self.type)
		image_list = os.listdir(image_dir)
		gt_list = os.listdir(gt_dir)
		image_filenames_int = [file.split('.')[0] for file in image_list if self.img_ext in file]
		gt_filenames_int = [file.split('.')[0] for file in gt_list if (self.gt_ext in file) and (self.task in file)]
		image_filenames_int.sort()
		gt_filenames_int.sort()

		# sanity check
		if not(image_filenames_int == gt_filenames_int):
			raise ValueError('image_filenames and gt_filenames do not match.')
		else:
			if self.max_examples is not None:
				if self.random_sample:
					import random
					random.shuffle(image_filenames_int)
				image_filenames_int = image_filenames_int[:self.max_examples]
			self.files = [name for name in image_filenames_int]

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		# load Image
		img_file = os.path.join(self.root, 'data', 'images', self.type, self.files[index] + self.img_ext)
		if not os.path.exists(img_file):
			raise ValueError('Cannot find image by path :' + img_file)
		img = load_image_with_cache_multicue_crops(img_file, cache=None) #self.cache)
		# load gt image
		gt_file = os.path.join(self.root, 'data', 'groundTruth', self.type, self.files[index] + '.' + self.task + self.gt_ext)
		gt = load_image_with_cache_multicue_crops(gt_file, cache=None, npy=True) #self.cache, matfile=True)
		if not self.return_filename:
			return self.transform(img, gt)
		else:
			return self.transform(img, gt), self.files[index] + self.img_ext

	def transform(self, img, gt):
		if len(gt.shape) == 3:
			gt = gt[:, :, 0]
		# gt_mean /= 255.
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
			_, h, w = gt.shape
			assert(self.crop_size < h and self.crop_size < w)
			i = random.randint(0, h - self.crop_size)
			j = random.randint(0, w - self.crop_size)
			img = img[:, i:i+self.crop_size, j:j+self.crop_size]
			gt = gt[:, i:i+self.crop_size, j:j+self.crop_size]
		return img, gt

def load_image_with_cache_tilt_illusion(path, cache=None):
	if cache is not None:
		if not cache.has_key(path):
			with open(path, 'rb') as f:
				cache[path] = f.read()
		return scipy.misc.imread(StringIO(cache[path]))
		# return Image.open(StringIO(cache[path]))
	else:
		return scipy.misc.imread(path)

class Tilt_illusion(data.Dataset):
	def __init__(self, root, type, test_mode=False,
		crop_size=None, rgb=True, scale=None,
		max_examples=None):
		self.root = root
		self.type = type # train or test
		self.test_mode=test_mode # if True, returns rich metadata
		self.crop_size = crop_size
		self.rgb = rgb
		self.scale = scale
		self.cache = {}

		# get list of images and gts from a specified path
		self.metadata = np.reshape(np.load(os.path.join(self.root, self.type, 'metadata', '1.npy')), (-1,11))
		self.image_dir = os.path.join(self.root, self.type, 'imgs')
		self.max_examples = max_examples

		if self.metadata is not None:
			self.metadata=self.metadata[:self.max_examples,:]

		self.gt = self.metadata[:,4].astype(np.float)
		self.gt_trig = np.concatenate((np.expand_dims(np.sin(np.pi * self.gt / 180.),axis=0),
									   np.expand_dims(np.cos(np.pi * self.gt / 180.),axis=0)
									   ),axis=0)
		# self.__getitem__(40)

	def __len__(self):
		return self.metadata.shape[0]

	def __getitem__(self, index):
		# load Image
		img_file = os.path.join(self.image_dir, '1', self.metadata[index, 1])
		if not os.path.exists(img_file):
			raise ValueError('Cannot find image by path :' + img_file)
		img = load_image_with_cache_multicue_crops(img_file, cache=None) #self.cache)
		# load gt image
		gt = self.gt_trig[:,index]
		if self.test_mode:
			meta = self.metadata[index,3:].astype(np.float) #[r1, theta1, lambda1, shift1, r2 ....]
			return self.transform(img, gt, meta)
		else:
			return self.transform(img, gt)

	def transform(self, img, gt, meta=None):
		gt = torch.from_numpy(np.array([gt])).float()

		img = np.array(img, dtype=np.float32)-127
		if self.scale is not None:
			img = cv2.resize(img, None, fx=self.scale[0], fy=self.scale[0], interpolation=cv2.INTER_LINEAR)
		img -= 127.
		img = np.transpose(np.tile(img, (3,1,1)),(1,2,0))
		data = []
		img = img.transpose((2, 0, 1))
		img = torch.from_numpy(img.copy()).float()
		if self.crop_size:
			_, h, w = img.shape
			assert(self.crop_size < h and self.crop_size < w)
			i = random.randint(0, h - self.crop_size)
			j = random.randint(0, w - self.crop_size)
			img = img[:, i:i+self.crop_size, j:j+self.crop_size]
		if meta is not None:
			return img, gt, meta
		else:
			return img, gt
