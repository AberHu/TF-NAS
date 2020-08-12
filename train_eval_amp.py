import os
import sys
import time
import glob
import logging
import argparse
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from apex.parallel import DistributedDataParallel as DDP
from apex import amp, parallel

from tools.utils import AverageMeter, accuracy
from tools.utils import count_parameters_in_MB
from tools.utils import create_exp_dir, save_checkpoint
from models.model_eval import Network, NetworkCfg
from parsing_model import get_op_and_depth_weights
from parsing_model import parse_architecture
from parsing_model import get_mc_num_dddict
from dataset import ImageList, pil_loader, cv2_loader
from dataset import IMAGENET_MEAN, IMAGENET_STD


parser = argparse.ArgumentParser("training the searched architecture on imagenet")
# various path
parser.add_argument('--train_root', type=str, required=True, help='training image root path')
parser.add_argument('--val_root', type=str, required=True, help='validating image root path')
parser.add_argument('--train_list', type=str, required=True, help='training image list')
parser.add_argument('--val_list', type=str, required=True, help='validating image list')
parser.add_argument('--model_path', type=str, default='', help='the searched model path')
parser.add_argument('--config_path', type=str, default='', help='the model config path')
parser.add_argument('--save', type=str, default='./checkpoints/', help='model and log saving path')
parser.add_argument('--snapshot', type=str, default='', help='for reset')

# training hyper-parameters
parser.add_argument('--print_freq', type=float, default=100, help='print frequency')
parser.add_argument('--workers', type=int, default=16, help='number of workers to load dataset')
parser.add_argument('--epochs', type=int, default=250, help='num of total training epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.2, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--num_classes', type=int, default=1000, help='class number of training set')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
parser.add_argument('--drop_connect_rate', type=float, default=0.2, help='dropout connect rate')

# amp and DDP hyper-parameters
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN')
parser.add_argument('--opt_level', type=str, default=None)
parser.add_argument('--keep_batchnorm_fp32', type=str, default=None)
parser.add_argument('--loss_scale', type=str, default=None)
parser.add_argument('--channels_last', type=bool, default=False)

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')


args, unparsed = parser.parse_known_args()

args.save = os.path.join(args.save, 'eval-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.note))
if args.local_rank == 0:
	create_exp_dir(args.save, scripts_to_save=None)

	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
		format=log_format, datefmt='%m/%d %I:%M:%S %p')
	fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
	fh.setFormatter(logging.Formatter(log_format))
	logging.getLogger().addHandler(fh)

if hasattr(torch, 'channels_last') and  hasattr(torch, 'contiguous_format'):
	if args.channels_last:
		memory_format = torch.channels_last
	else:
		memory_format = torch.contiguous_format
else:
	memory_format = None


class CrossEntropyLabelSmooth(nn.Module):
	def __init__(self, num_classes, epsilon):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, xs, targets):
		log_probs = self.logsoftmax(xs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (-targets * log_probs).mean(0).sum()
		return loss


def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)


def main():
	if not torch.cuda.is_available():
		logging.info('No GPU device available')
		sys.exit(1)
	set_seed(args.seed)
	cudnn.enabled=True
	cudnn.benchmark = True
	args.distributed = False
	if 'WORLD_SIZE' in os.environ:
		args.distributed = int(os.environ['WORLD_SIZE']) > 1
	args.gpu = 0
	args.world_size = 1
	if args.distributed:
		set_seed(args.local_rank)
		args.gpu = args.local_rank
		torch.cuda.set_device(args.gpu)
		torch.distributed.init_process_group(backend='nccl', init_method='env://')
		args.world_size = torch.distributed.get_world_size()
	if args.local_rank == 0:
		logging.info("args = {}".format(args))
		logging.info("unparsed_args = {}".format(unparsed))
		logging.info("distributed = {}".format(args.distributed))
		logging.info("opt_level = {}".format(args.opt_level))
		logging.info("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32))
		logging.info("loss_scale = {}".format(args.loss_scale))
		logging.info("CUDNN VERSION: {}".format(torch.backends.cudnn.version()))

	# create model
	if args.local_rank == 0:
		logging.info('parsing the architecture')
	if args.model_path and os.path.isfile(args.model_path):
		op_weights, depth_weights = get_op_and_depth_weights(args.model_path)
		parsed_arch = parse_architecture(op_weights, depth_weights)
		mc_mask_dddict = torch.load(args.model_path)['mc_mask_dddict']
		mc_num_dddict  = get_mc_num_dddict(mc_mask_dddict)
		model = Network(args.num_classes, parsed_arch, mc_num_dddict, None, args.dropout_rate, args.drop_connect_rate)
	elif args.config_path and os.path.isfile(args.config_path):
		model_config = json.load(open(args.config_path, 'r'))
		model = NetworkCfg(args.num_classes, model_config, None, args.dropout_rate, args.drop_connect_rate)
	else:
		raise Exception('invalid --model_path and --config_path')
	if args.sync_bn:
		if args.local_rank == 0: logging.info("using apex synced BN")
		model = parallel.convert_syncbn_model(model)
	model = model.cuda().to(memory_format=memory_format) if memory_format is not None else model.cuda()
	config = model.config
	if args.local_rank == 0:
		with open(os.path.join(args.save, 'model.config'), 'w') as f:
			json.dump(config, f, indent=4)
		# logging.info(config)
		logging.info("param size = %fMB", count_parameters_in_MB(model))

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.cuda()
	criterion_smooth = CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)
	criterion_smooth = criterion_smooth.cuda()

	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)
	# Initialize Amp
	if args.opt_level is not None:
		model, optimizer = amp.initialize(model, optimizer,
										  opt_level=args.opt_level,
										  keep_batchnorm_fp32=args.keep_batchnorm_fp32,
										  loss_scale=args.loss_scale)

	# For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
	# This must be done AFTER the call to amp.initialize.
	if args.distributed:
		# By default, apex.parallel.DistributedDataParallel overlaps communication with
		# computation in the backward pass.
		# delay_allreduce delays all communication to the end of the backward pass.
		model = DDP(model, delay_allreduce=True)
	else:
		model = nn.DataParallel(model)

	# define transform and initialize dataloader
	batch_size = args.batch_size // args.world_size
	workers    = args.workers    // args.world_size
	train_transform = transforms.Compose([
							transforms.RandomResizedCrop(224),
							transforms.RandomHorizontalFlip(),
							transforms.ColorJitter(
								brightness=0.4,
								contrast=0.4,
								saturation=0.4,#),
								hue=0.2),
							transforms.ToTensor(),
							transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
						])
	val_transform   = transforms.Compose([
							transforms.Resize(256),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
							transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
						])
	train_dataset = ImageList(root=args.train_root, 
							  list_path=args.train_list, 
							  transform=train_transform)
	val_dataset   = ImageList(root=args.val_root, 
							  list_path=args.val_list, 
							  transform=val_transform)
	train_sampler = None
	val_sampler   = None
	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
		val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset)
	train_loader = torch.utils.data.DataLoader(
						train_dataset, batch_size=batch_size, num_workers=workers,
						pin_memory=True, sampler=train_sampler, shuffle=(train_sampler is None))
	val_loader   = torch.utils.data.DataLoader(
						val_dataset, batch_size=batch_size, num_workers=workers, 
						pin_memory=True, sampler=val_sampler, shuffle=False)

	# define learning rate scheduler
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
	best_acc_top1 = 0
	best_acc_top5 = 0
	start_epoch = 0

	# restart from snapshot
	if args.snapshot and os.path.isfile(args.snapshot):
		if args.local_rank == 0:
			logging.info('loading snapshot from {}'.format(args.snapshot))
		checkpoint = torch.load(args.snapshot, map_location=lambda storage,loc: storage.cuda(args.gpu))
		start_epoch = checkpoint['epoch']
		best_acc_top1 = checkpoint['best_acc_top1']
		best_acc_top5 = checkpoint['best_acc_top5']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		if args.opt_level is not None:
			amp.load_state_dict(checkpoint['amp'])
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), last_epoch=0)
		for epoch in range(start_epoch):
			current_lr = scheduler.get_lr()[0]
			if args.local_rank == 0:
				logging.info('Epoch: %d lr %e', epoch, current_lr)
			if epoch < 5 and args.batch_size > 256:
				for param_group in optimizer.param_groups:
					param_group['lr'] = current_lr * (epoch + 1) / 5.0
				if args.local_rank == 0:
					logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
			if epoch < 5 and args.batch_size > 256:
				for param_group in optimizer.param_groups:
					param_group['lr'] = current_lr
			scheduler.step()

	# the main loop
	for epoch in range(start_epoch, args.epochs):
		current_lr = scheduler.get_lr()[0]
		if args.local_rank == 0:
			logging.info('Epoch: %d lr %e', epoch, current_lr)
		if epoch < 5 and args.batch_size > 256:
			for param_group in optimizer.param_groups:
				param_group['lr'] = current_lr * (epoch + 1) / 5.0
			if args.local_rank == 0:
				logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)

		if args.distributed:
			train_sampler.set_epoch(epoch)

		epoch_start = time.time()
		train_acc, train_obj = train(train_loader, model, criterion_smooth, optimizer)
		if args.local_rank == 0:
			logging.info('Train_acc: %f', train_acc)

		val_acc_top1, val_acc_top5, val_obj = validate(val_loader, model, criterion)
		if args.local_rank == 0:
			logging.info('Val_acc_top1: %f', val_acc_top1)
			logging.info('Val_acc_top5: %f', val_acc_top5)
			logging.info('Epoch time: %ds.', time.time() - epoch_start)

		if args.local_rank == 0:
			is_best = False
			if val_acc_top1 > best_acc_top1:
				best_acc_top1 = val_acc_top1
				best_acc_top5 = val_acc_top5
				is_best = True
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'best_acc_top1': best_acc_top1,
				'best_acc_top5': best_acc_top5,
				'optimizer' : optimizer.state_dict(),
				'amp': amp.state_dict() if args.opt_level is not None else None,
				}, is_best, args.save)

		if epoch < 5 and args.batch_size > 256:
			for param_group in optimizer.param_groups:
				param_group['lr'] = current_lr

		scheduler.step()


def train(train_loader, model, criterion, optimizer):
	objs = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	model.train()

	end = time.time()
	for step, data in enumerate(train_loader):
		data_time.update(time.time() - end)
		x = data[0].cuda(non_blocking=True)
		target = data[1].cuda(non_blocking=True)

		# forward
		batch_start = time.time()
		logits = model(x)
		loss = criterion(logits, target)

		# backward
		optimizer.zero_grad()
		if args.opt_level is not None:
			with amp.scale_loss(loss, optimizer) as scaled_loss:
				scaled_loss.backward()
		else:
			loss.backward()
		if args.grad_clip > 0:
			torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.grad_clip)
		optimizer.step()
		batch_time.update(time.time() - batch_start)

		if step % args.print_freq == 0:
			# For better performance, don't accumulate these metrics every iteration,
			# since they may incur an allreduce and some host<->device syncs.
			prec1, prec5 = accuracy(logits, target, topk=(1, 5))
			if args.distributed:
				reduced_loss = reduce_tensor(loss.data)
				prec1 = reduce_tensor(prec1)
				prec5 = reduce_tensor(prec5)
			else:
				reduced_loss = loss.data
			objs.update(reduced_loss.item(), x.size(0))
			top1.update(prec1.item(), x.size(0))
			top5.update(prec5.item(), x.size(0))
			torch.cuda.synchronize()

			duration = 0 if step == 0 else time.time() - duration_start
			duration_start = time.time()
			if args.local_rank == 0:
				logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs DTime: %.4fs', 
										step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg, data_time.avg)
		end = time.time()

	return top1.avg, objs.avg


def validate(val_loader, model, criterion):
	objs = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	model.eval()

	for step, data in enumerate(val_loader):
		x = data[0].cuda(non_blocking=True)
		target = data[1].cuda(non_blocking=True)

		with torch.no_grad():
			logits = model(x)
			loss = criterion(logits, target)

		prec1, prec5 = accuracy(logits, target, topk=(1, 5))
		if args.distributed:
			reduced_loss = reduce_tensor(loss.data)
			prec1 = reduce_tensor(prec1)
			prec5 = reduce_tensor(prec5)
		else:
			reduced_loss = loss.data
		objs.update(reduced_loss.item(), x.size(0))
		top1.update(prec1.item(), x.size(0))
		top5.update(prec5.item(), x.size(0))

		if args.local_rank == 0 and step % args.print_freq == 0:
			duration = 0 if step == 0 else time.time() - duration_start
			duration_start = time.time()
			logging.info('VALIDATE Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg, duration)

	return top1.avg, top5.avg, objs.avg


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
	main()
