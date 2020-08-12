import os
import sys
import time
import glob
import logging
import argparse
import json
import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from tools.utils import AverageMeter, accuracy
from tools.utils import count_parameters_in_MB
from tools.utils import create_exp_dir, save_checkpoint
from models.model_eval import Network, NetworkCfg
from parsing_model import get_op_and_depth_weights
from parsing_model import parse_architecture
from parsing_model import get_mc_num_dddict
from dataset import ImageList, pil_loader, cv2_loader
from dataset import IMAGENET_MEAN, IMAGENET_STD


parser = argparse.ArgumentParser("testing the trained architectures")
# various path
parser.add_argument('--val_root', type=str, required=True, help='validating image root path')
parser.add_argument('--val_list', type=str, required=True, help='validating image list')
parser.add_argument('--model_path', type=str, default='', help='the searched model path')
parser.add_argument('--config_path', type=str, default='', help='the model config path')
parser.add_argument('--weights', type=str, required=True, help='pretrained model weights')

# training hyper-parameters
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--num_classes', type=int, default=1000, help='class number of training set')


args, unparsed = parser.parse_known_args()


def main():
	if not torch.cuda.is_available():
		print('No GPU device available')
		sys.exit(1)
	cudnn.enabled=True
	cudnn.benchmark = True

	# create model
	print('parsing the architecture')
	if args.model_path and os.path.isfile(args.model_path):
		op_weights, depth_weights = get_op_and_depth_weights(args.model_path)
		parsed_arch = parse_architecture(op_weights, depth_weights)
		mc_mask_dddict = torch.load(args.model_path)['mc_mask_dddict']
		mc_num_dddict  = get_mc_num_dddict(mc_mask_dddict)
		model = Network(args.num_classes, parsed_arch, mc_num_dddict, None, 0.0, 0.0)
	elif args.config_path and os.path.isfile(args.config_path):
		model_config = json.load(open(args.config_path, 'r'))
		model = NetworkCfg(args.num_classes, model_config, None, 0.0, 0.0)
	else:
		raise Exception('invalid --model_path and --config_path')
	model = nn.DataParallel(model).cuda()

	# load pretrained weights
	if os.path.exists(args.weights) and os.path.isfile(args.weights):
		print('loading weights from {}'.format(args.weights))
		checkpoint = torch.load(args.weights)
		model.load_state_dict(checkpoint['state_dict'])

	# define transform and initialize dataloader
	val_transform   = transforms.Compose([
							transforms.Resize(256),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
							transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
						])
	val_queue   = torch.utils.data.DataLoader(
		ImageList(root=args.val_root, 
				  list_path=args.val_list, 
				  transform=val_transform,), 
		batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

	start = time.time()
	val_acc_top1, val_acc_top5 = validate(val_queue, model)
	print('Val_acc_top1: {:.2f}'.format(val_acc_top1))
	print('Val_acc_top5: {:.2f}'.format(val_acc_top5))
	print('Test time: %ds.', time.time() - start)


def validate(val_queue, model):
	top1 = AverageMeter()
	top5 = AverageMeter()
	model.eval()

	for data in tqdm.tqdm(val_queue):
		x = data[0].cuda(non_blocking=True)
		target = data[1].cuda(non_blocking=True)

		with torch.no_grad():
			logits = model(x)

		prec1, prec5 = accuracy(logits, target, topk=(1, 5))
		n = x.size(0)
		top1.update(prec1.data.item(), n)
		top5.update(prec5.data.item(), n)

	return top1.avg, top5.avg


if __name__ == '__main__':
	main()
