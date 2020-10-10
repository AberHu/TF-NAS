import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import pickle

sys.path.append('..')

from tools.utils import measure_latency_in_ms
from models.layers import *

cudnn.enabled = True
cudnn.benchmark = True


PRIMITIVES = [
	'MBI_k3_e4',
	'MBI_k3_e8',
	'MBI_k5_e4',
	'MBI_k5_e8',
	'MBI_k3_e4_se',
	'MBI_k3_e8_se',
	'MBI_k5_e4_se',
	'MBI_k5_e8_se',
	# 'skip',
]

OPS = {
	'MBI_k3_e4' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
	'MBI_k3_e8' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
	'MBI_k5_e4' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
	'MBI_k5_e8' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
	'MBI_k3_e4_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 3, s, affine=aff, act_func=act),
	'MBI_k3_e8_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic*2, oc, 3, s, affine=aff, act_func=act),
	'MBI_k5_e4_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 5, s, affine=aff, act_func=act),
	'MBI_k5_e8_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic*2, oc, 5, s, affine=aff, act_func=act),
	# 'skip'      : lambda ic, mc, oc, s, aff, act: IdentityLayer(ic, oc),
}


def get_latency_lookup(is_cuda):
	latency_lookup = OrderedDict()

	# first 3x3 conv, 3x3 sep conv, last 1x1 conv, avgpool, fc
	print('first 3x3 conv, 3x3 sep conv, last 1x1 conv, avgpool, fc')
	block = ConvLayer(3, 32, kernel_size=3, stride=2, affine=True, act_func='relu')
	shape = (32, 3, 224, 224) if is_cuda else (1, 3, 224, 224)
	lat1  = measure_latency_in_ms(block, shape, is_cuda)
	# time.sleep(0.1)
	block = MBInvertedResBlock(32, 32, 8, 16, kernel_size=3, stride=1, affine=True, act_func='relu')
	shape = (32, 32, 112, 112) if is_cuda else (1, 32, 112, 112)
	lat2  = measure_latency_in_ms(block, shape, is_cuda)
	# time.sleep(0.1)
	block = ConvLayer(320, 1280, kernel_size=1, stride=1, affine=True, act_func='swish')
	shape = (32, 320, 7, 7) if is_cuda else (1, 320, 7, 7)
	lat3  = measure_latency_in_ms(block, shape, is_cuda)
	# time.sleep(0.1)
	block = nn.AdaptiveAvgPool2d(1)
	shape = (32, 1280, 7, 7) if is_cuda else (1, 1280, 7, 7)
	lat4  = measure_latency_in_ms(block, shape, is_cuda)
	# time.sleep(0.1)
	block = LinearLayer(1280, 1000)
	shape = (32, 1280) if is_cuda else (1, 1280)
	lat5  = measure_latency_in_ms(block, shape, is_cuda)
	# time.sleep(0.1)
	latency_lookup['base'] = lat1 + lat2 + lat3 + lat4 + lat5 # + 0.1  # 0.1 is the latency rectifier


	# 112x112 cin=16 cout=24 s=2 relu
	print('112x112 cin=16 cout=24 s=2 relu')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 16*4+1))
			# mc_list = list(range(0, 16*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 16*8+1))
			# mc_list = list(range(0, 16*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 16*2+1))
			# mc_list = list(range(0, 16*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](16, mc, 24, 2, True, 'relu')
			shape = (32, 16, 112, 112) if is_cuda else (1, 16, 112, 112)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_112_16_0_24_k{}_s2_relu'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_112_16_16_24_k{}_s2_relu'.format(block.name, block.kernel_size)
				else:
					key = '{}_112_16_32_24_k{}_s2_relu'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)


	# 56x56 cin=24 cout=24 s=1 relu
	print('56x56 cin=24 cout=24 s=1 relu')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 24*4+1))
			# mc_list = list(range(0, 24*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 24*8+1))
			# mc_list = list(range(0, 24*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 24*2+1))
			# mc_list = list(range(0, 24*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](24, mc, 24, 1, True, 'relu')
			shape = (32, 24, 56, 56) if is_cuda else (1, 24, 56, 56)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_56_24_0_24_k{}_s1_relu'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_56_24_24_24_k{}_s1_relu'.format(block.name, block.kernel_size)
				else:
					key = '{}_56_24_48_24_k{}_s1_relu'.format(block.name, block.kernel_size)	
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)


	# 56x56 cin=24 cout=40 s=2 swish
	print('56x56 cin=24 cout=40 s=2 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 24*4+1))
			# mc_list = list(range(0, 24*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 24*8+1))
			# mc_list = list(range(0, 24*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 24*2+1))
			# mc_list = list(range(0, 24*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](24, mc, 40, 2, True, 'swish')
			shape = (32, 24, 56, 56) if is_cuda else (1, 24, 56, 56)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_56_24_0_40_k{}_s2_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_56_24_24_40_k{}_s2_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_56_24_48_40_k{}_s2_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	# 28x28 cin=40 cout=40 s=1 swish
	print('28x28 cin=40 cout=40 s=1 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 40*4+1))
			# mc_list = list(range(0, 40*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 40*8+1))
			# mc_list = list(range(0, 40*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 40*2+1))
			# mc_list = list(range(0, 40*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](40, mc, 40, 1, True, 'swish')
			shape = (32, 40, 28, 28) if is_cuda else (1, 40, 28, 28)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_28_40_0_40_k{}_s1_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_28_40_40_40_k{}_s1_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_28_40_80_40_k{}_s1_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	# 28x28 cin=40 cout=80 s=2 swish
	print('28x28 cin=40 cout=80 s=2 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 40*4+1))
			# mc_list = list(range(0, 40*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 40*8+1))
			# mc_list = list(range(0, 40*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 40*2+1))
			# mc_list = list(range(0, 40*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](40, mc, 80, 2, True, 'swish')
			shape = (32, 40, 28, 28) if is_cuda else (1, 40, 28, 28)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_28_40_0_80_k{}_s2_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_28_40_40_80_k{}_s2_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_28_40_80_80_k{}_s2_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	# 14x14 cin=80 cout=80 s=1 swish
	print('14x14 cin=80 cout=80 s=1 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 80*4+1))
			# mc_list = list(range(0, 80*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 80*8+1))
			# mc_list = list(range(0, 80*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 80*2+1))
			# mc_list = list(range(0, 80*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](80, mc, 80, 1, True, 'swish')
			shape = (32, 80, 14, 14) if is_cuda else (1, 80, 14, 14)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_14_80_0_80_k{}_s1_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 ==0:
					key = '{}_14_80_80_80_k{}_s1_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_14_80_160_80_k{}_s1_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	# 14x14 cin=80 cout=112 s=1 swish
	print('14x14 cin=80 cout=112 s=1 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 80*4+1))
			# mc_list = list(range(0, 80*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 80*8+1))
			# mc_list = list(range(0, 80*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 80*2+1))
			# mc_list = list(range(0, 80*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](80, mc, 112, 1, True, 'swish')
			shape = (32, 80, 14, 14) if is_cuda else (1, 80, 14, 14)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_14_80_0_112_k{}_s1_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_14_80_80_112_k{}_s1_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_14_80_160_112_k{}_s1_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	# 14x14 cin=112 cout=112 s=1 swish
	print('14x14 cin=112 cout=112 s=1 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 112*4+1))
			# mc_list = list(range(0, 112*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 112*8+1))
			# mc_list = list(range(0, 112*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 112*2+1))
			# mc_list = list(range(0, 112*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](112, mc, 112, 1, True, 'swish')
			shape = (32, 112, 14, 14) if is_cuda else (1, 112, 14, 14)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_14_112_0_112_k{}_s1_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_14_112_112_112_k{}_s1_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_14_112_224_112_k{}_s1_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	# 14x14 cin=112 cout=192 s=2 swish
	print('14x14 cin=112 cout=192 s=2 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 112*4+1))
			# mc_list = list(range(0, 112*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 112*8+1))
			# mc_list = list(range(0, 112*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 112*2+1))
			# mc_list = list(range(0, 112*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](112, mc, 192, 2, True, 'swish')
			shape = (32, 112, 14, 14) if is_cuda else (1, 112, 14, 14)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_14_112_0_192_k{}_s2_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_14_112_112_192_k{}_s2_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_14_112_224_192_k{}_s2_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	# 7x7 cin=192 cout=192 s=1 swish
	print('7x7 cin=192 cout=192 s=1 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 192*4+1))
			# mc_list = list(range(0, 192*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 192*8+1))
			# mc_list = list(range(0, 192*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 192*2+1))
			# mc_list = list(range(0, 192*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](192, mc, 192, 1, True, 'swish')
			shape = (32, 192, 7, 7) if is_cuda else (1, 192, 7, 7)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_7_192_0_192_k{}_s1_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_7_192_192_192_k{}_s1_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_7_192_384_192_k{}_s1_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	# 7x7 cin=192 cout=320 s=1 swish
	print('7x7 cin=192 cout=320 s=1 swish')
	for idx in range(len(PRIMITIVES)):
		if (idx == 0) or (idx == 2):
			continue

		op = PRIMITIVES[idx]
		if op.startswith('MBI') and (idx % 2 == 0):
			mc_list = list(range(1, 192*4+1))
			# mc_list = list(range(0, 192*4+1, 8))
			# mc_list[0] = 1
		elif op.startswith('MBI') and (idx % 2 == 1):
			mc_list = list(range(1, 192*8+1))
			# mc_list = list(range(0, 192*8+1, 8))
			# mc_list[0] = 1
		elif op.startswith('Bot'):
			mc_list = list(range(1, 192*2+1))
			# mc_list = list(range(0, 192*2+1, 8))
			# mc_list[0] = 1
		else:
			raise ValueError

		for mc in mc_list:
			block = OPS[PRIMITIVES[idx]](192, mc, 320, 1, True, 'swish')
			shape = (32, 192, 7, 7) if is_cuda else (1, 192, 7, 7)
			lat   = measure_latency_in_ms(block, shape, is_cuda)
			if idx < 4:
				key = '{}_7_192_0_320_k{}_s1_swish'.format(block.name, block.kernel_size)
			else:
				if idx % 2 == 0:
					key = '{}_7_192_192_320_k{}_s1_swish'.format(block.name, block.kernel_size)
				else:
					key = '{}_7_192_384_320_k{}_s1_swish'.format(block.name, block.kernel_size)
			if key not in latency_lookup:
				latency_lookup[key] = OrderedDict()
			latency_lookup[key][block.mid_channels] = lat
			# time.sleep(0.1)

	return latency_lookup


# def convert_latency_lookup(latency_lookup):
# 	new_latency_lookup = OrderedDict()

# 	for key in latency_lookup:
# 		if key == 'base':
# 			new_latency_lookup['base'] = latency_lookup['base']
# 		else:
# 			mc_list  = list(latency_lookup[key].keys())
# 			lat_list = sorted(list(latency_lookup[key].values()))
# 			new_mc_list  = []
# 			new_lat_list = []
# 			for new_mc in range(1, mc_list[-1]+1):
# 				for idx in range(len(mc_list)):
# 					if new_mc == mc_list[idx]:
# 						new_mc_list.append(new_mc)
# 						new_lat_list.append(lat_list[idx])
# 						break
# 					if new_mc < mc_list[idx]:
# 						new_mc_list.append(new_mc)
# 						interval = (lat_list[idx] - lat_list[idx-1]) / (mc_list[idx] - mc_list[idx-1])
# 						new_lat = (new_mc - mc_list[idx-1]) * interval + lat_list[idx-1]
# 						new_lat_list.append(new_lat)
# 						break
# 			new_latency_lookup[key] = OrderedDict(list(zip(new_mc_list, new_lat_list)))

# 	return new_latency_lookup


if __name__ == '__main__':
	print('measure latency on gpu......')
	latency_lookup = get_latency_lookup(is_cuda=True)
	# latency_lookup = convert_latency_lookup(latency_lookup)
	with open('latency_gpu_example.pkl', 'wb') as f:
		pickle.dump(latency_lookup, f)

	print('measure latency on cpu......')
	latency_lookup = get_latency_lookup(is_cuda=False)
	# latency_lookup = convert_latency_lookup(latency_lookup)
	with open('latency_cpu_example.pkl', 'wb') as f:
		pickle.dump(latency_lookup, f)
