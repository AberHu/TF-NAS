import os
import sys
import argparse
import pickle
import json
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models.model_eval import Network
from tools.utils import measure_latency_in_ms, count_parameters_in_MB
from tools.flops_benchmark import calculate_FLOPs_in_M


cudnn.enabled = True
cudnn.benchmark = True


def get_op_and_depth_weights(model_or_path):
	if isinstance(model_or_path, str):  # for model_path
		checkpoint = torch.load(model_or_path)
		state_dict = checkpoint['state_dict']
	else:                               # for model
		state_dict = model_or_path.state_dict()

	op_weights = []
	depth_weights = []

	for key in state_dict:
		if key.endswith('log_alphas'):
			op_weights.append(np.exp(state_dict[key].cpu().numpy()))
		elif key.endswith('betas'):
			depth_weights.append(F.softmax(state_dict[key].cpu(), dim=-1).numpy())
		else:
			continue

	return op_weights, depth_weights


def parse_architecture(op_weights, depth_weights):
	parsed_arch = OrderedDict([
			('stage1', OrderedDict([('block1', -1), ('block2', -1)])),
			('stage2', OrderedDict([('block1', -1), ('block2', -1), ('block3', -1)])),
			('stage3', OrderedDict([('block1', -1), ('block2', -1), ('block3', -1), ('block4', -1)])),
			('stage4', OrderedDict([('block1', -1), ('block2', -1), ('block3', -1), ('block4', -1)])),
			('stage5', OrderedDict([('block1', -1), ('block2', -1), ('block3', -1), ('block4', -1)])),
			('stage6', OrderedDict([('block1', -1)])),
		])

	stages = []
	blocks = []
	for stage in parsed_arch:
		for block in parsed_arch[stage]:
			stages.append(stage)
			blocks.append(block)

	op_max_indexes = [np.argmax(x) for x in op_weights]
	for (stage, block, op_max_index) in zip(stages, blocks, op_max_indexes):
		parsed_arch[stage][block] = op_max_index

	depth_max_indexes = [np.argmax(x)+1 for x in depth_weights]
	for stage_index, depth_max_index in enumerate(depth_max_indexes, start=1):
		stage = 'stage{}'.format(stage_index)
		for block_index in range(depth_max_index+1, 5+1):
			block = 'block{}'.format(block_index)
			if block in parsed_arch[stage]:
				del parsed_arch[stage][block]

	return parsed_arch


def get_mc_num_dddict(mc_mask_dddict, is_max=False):
	mc_num_dddict = OrderedDict()
	for stage in mc_mask_dddict:
		mc_num_dddict[stage] = OrderedDict()
		for block in mc_mask_dddict[stage]:
			mc_num_dddict[stage][block] = OrderedDict()
			for op_idx in mc_mask_dddict[stage][block]:
				if is_max:
					mc_num_dddict[stage][block][op_idx] = mc_mask_dddict[stage][block][op_idx].size(0)
				else:
					mc_num_dddict[stage][block][op_idx] = int(sum(mc_mask_dddict[stage][block][op_idx]).item())

	return mc_num_dddict


if __name__ == '__main__':
	parser = argparse.ArgumentParser("parsing TF-NAS")
	parser.add_argument('--model_path', type=str, required=True, help='path of searched model')
	parser.add_argument('--save_path', type=str, default='.', help='saving path of parsed architecture config')
	parser.add_argument('--lookup_path', type=str, default='../latency_pkl/latency_gpu.pkl', help='path of latency lookup')
	parser.add_argument('--print_lat', action='store_true', help='measure and print the latency')

	args = parser.parse_args()

	op_weights, depth_weights = get_op_and_depth_weights(args.model_path)
	parsed_arch = parse_architecture(op_weights, depth_weights)
	with open(args.lookup_path, 'rb') as f:
		lat_lookup = pickle.load(f)
	mc_mask_dddict = torch.load(args.model_path)['mc_mask_dddict']
	mc_num_dddict  = get_mc_num_dddict(mc_mask_dddict)
	model = Network(1000, parsed_arch, mc_num_dddict, lat_lookup, 0.0, 0.0)
	model = model.cuda()

	x = torch.randn((1, 3, 224, 224))
	x = x.cuda()

	config = model.config
	with open(args.save_path, 'w') as f:
		json.dump(config, f, indent=4)

	params = count_parameters_in_MB(model)
	print('Params:  \t{:.4f}MB'.format(params))

	flops = calculate_FLOPs_in_M(model, (1, 3, 224, 224))
	print('FLOPs:  \t{:.4f}M'.format(flops))

	if args.print_lat:
		# latency in lookup table
		lat_lut = model.get_lookup_latency(x)
		print('Lat_LUT:\t{:.4f}ms'.format(lat_lut))

		lat_gpu = measure_latency_in_ms(model, (32, 3, 224, 224), is_cuda=True)
		print('Lat_GPU bs=32:\t{:.4f}ms'.format(lat_gpu))

		lat_gpu = measure_latency_in_ms(model, (1, 3, 224, 224), is_cuda=True)
		print('Lat_GPU bs=1:\t{:.4f}ms'.format(lat_gpu))

		lat_cpu = measure_latency_in_ms(model, (1, 3, 224, 224), is_cuda=False)
		print('Lat_CPU bs=1:\t{:.4f}ms'.format(lat_cpu))