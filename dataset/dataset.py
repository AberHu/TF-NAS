import os
import cv2
from PIL import Image
import torch.utils.data as data
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# If get UserWarning: Corrupt EXIF data, use cv2_loader or ignore warnings
def pil_loader(path):
	img = Image.open(path).convert('RGB')
	return img

def cv2_loader(path):
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = Image.fromarray(img)
	return img

default_loader = pil_loader


def default_list_reader(list_path):
	img_list = []
	with open(list_path, 'r') as f:
		for line in f.readlines():
			img_path, label = line.strip().split(' ')
			img_list.append((img_path, int(label)))

	return img_list


class ImageList(data.Dataset):
	def __init__(self, root, list_path, transform=None, list_reader=default_list_reader, loader=default_loader):
		self.root       = root
		self.img_list   = list_reader(list_path)
		self.transform  = transform
		self.loader     = loader

	def __getitem__(self, index):
		img_path, target = self.img_list[index]
		img = self.loader(os.path.join(self.root, img_path))

		if self.transform:
			img = self.transform(img)

		return img, target

	def __len__(self):
		return len(self.img_list)
