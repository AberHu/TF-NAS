import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

PRIMITIVES = [
	'MBI_k3_e3',
	'MBI_k3_e6',
	'MBI_k5_e3',
	'MBI_k5_e6',
	'MBI_k3_e3_se',
	'MBI_k3_e6_se',
	'MBI_k5_e3_se',
	'MBI_k5_e6_se',
	# 'skip',
]

OPS = {
	'MBI_k3_e3' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
	'MBI_k3_e6' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 3, s, affine=aff, act_func=act),
	'MBI_k5_e3' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
	'MBI_k5_e6' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, 0, oc, 5, s, affine=aff, act_func=act),
	'MBI_k3_e3_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 3, s, affine=aff, act_func=act),
	'MBI_k3_e6_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic*2, oc, 3, s, affine=aff, act_func=act),
	'MBI_k5_e3_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic  , oc, 5, s, affine=aff, act_func=act),
	'MBI_k5_e6_se' : lambda ic, mc, oc, s, aff, act: MBInvertedResBlock(ic, mc, ic*2, oc, 5, s, affine=aff, act_func=act),
	# 'skip'      : lambda ic, mc, oc, s, aff, act: IdentityLayer(ic, oc),
}


class Network(nn.Module):
	def __init__(self, num_classes, parsed_arch, mc_num_dddict, lat_lookup=None, dropout_rate=0.0, drop_connect_rate=0.0):
		super(Network, self).__init__()
		self.lat_lookup = lat_lookup
		self.mc_num_dddict = mc_num_dddict
		self.parsed_arch = parsed_arch
		self.dropout_rate = dropout_rate
		self.drop_connect_rate = drop_connect_rate
		self.block_count = self._get_block_count()
		self.block_idx = 0

		self.first_stem  = ConvLayer(3, 32, kernel_size=3, stride=2, affine=True, act_func='relu')
		self.second_stem = MBInvertedResBlock(32, 32, 8, 16, kernel_size=3, stride=1, affine=True, act_func='relu')
		self.block_idx += 1
		self.second_stem.drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
		self.stage1 = self._make_stage('stage1',
										ics  = [16,24],
										ocs  = [24,24],
										ss   = [2,1],
										affs = [True, True],
										acts = ['relu', 'relu'],)
		self.stage2 = self._make_stage('stage2',
										ics  = [24,40,40],
										ocs  = [40,40,40],
										ss   = [2,1,1],
										affs = [True, True, True],
										acts = ['swish', 'swish', 'swish'],)
		self.stage3 = self._make_stage('stage3',
										ics  = [40,80,80,80],
										ocs  = [80,80,80,80],
										ss   = [2,1,1,1],
										affs = [True, True, True, True],
										acts = ['swish', 'swish', 'swish', 'swish'],)
		self.stage4 = self._make_stage('stage4',
										ics  = [80,112,112,112],
										ocs  = [112,112,112,112],
										ss   = [1,1,1,1],
										affs = [True, True, True, True],
										acts = ['swish', 'swish', 'swish', 'swish'],)
		self.stage5 = self._make_stage('stage5',
										ics  = [112,192,192,192],
										ocs  = [192,192,192,192],
										ss   = [2,1,1,1],
										affs = [True, True, True, True],
										acts = ['swish', 'swish', 'swish', 'swish'],)
		self.stage6 = self._make_stage('stage6',
										ics  = [192,],
										ocs  = [320,],
										ss   = [1,],
										affs = [True,],
										acts = ['swish',],)
		self.feature_mix_layer = ConvLayer(320, 1280, kernel_size=1, stride=1, affine=True, act_func='swish')
		self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
		self.classifier = LinearLayer(1280, num_classes)

		self._initialization()

	def _get_block_count(self):
		count = 1
		for stage in self.parsed_arch:
			count += len(self.parsed_arch[stage])

		return count

	def _make_stage(self, stage_name, ics, ocs, ss, affs, acts):
		stage = nn.ModuleList()
		for i, block_name in enumerate(self.parsed_arch[stage_name]):
			self.block_idx += 1
			op_idx = self.parsed_arch[stage_name][block_name]
			primitive = PRIMITIVES[op_idx]
			mc = self.mc_num_dddict[stage_name][block_name][op_idx]
			op = OPS[primitive](ics[i], mc, ocs[i], ss[i], affs[i], acts[i])
			op.drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
			stage.append(op)

		return stage

	def forward(self, x):
		x = self.first_stem(x)
		x = self.second_stem(x)

		for block in self.stage1:
			x = block(x)
		for block in self.stage2:
			x = block(x)
		for block in self.stage3:
			x = block(x)
		for block in self.stage4:
			x = block(x)
		for block in self.stage5:
			x = block(x)
		for block in self.stage6:
			x = block(x)

		x = self.feature_mix_layer(x)
		x = self.global_avg_pooling(x)
		x = x.view(x.size(0), -1)
		if self.dropout_rate > 0.0:
			x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.classifier(x)
		
		return x

	def get_lookup_latency(self, x):
		if not self.lat_lookup:
			return 0.0

		lat = self.lat_lookup['base']
		x = self.first_stem(x)
		x = self.second_stem(x)

		for block in self.stage1:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage2:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage3:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage4:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage5:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage6:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)

		return lat

	@property
	def config(self):
		return {
			'first_stem':  self.first_stem.config,
			'second_stem': self.second_stem.config,
			'stage1': [block.config for block in self.stage1],
			'stage2': [block.config for block in self.stage2],
			'stage3': [block.config for block in self.stage3],
			'stage4': [block.config for block in self.stage4],
			'stage5': [block.config for block in self.stage5],
			'stage6': [block.config for block in self.stage6],
			'feature_mix_layer': self.feature_mix_layer.config,
			'classifier': self.classifier.config,
		}

	def _initialization(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					nn.init.constant_(m.weight, 1)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)


class NetworkCfg(nn.Module):
	def __init__(self, num_classes, model_config, lat_lookup=None, dropout_rate=0.0, drop_connect_rate=0.0):
		super(NetworkCfg, self).__init__()
		self.lat_lookup = lat_lookup
		self.model_config = model_config
		self.dropout_rate = dropout_rate
		self.drop_connect_rate = drop_connect_rate
		self.block_count = self._get_block_count()
		self.block_idx = 0

		self.first_stem  = set_layer_from_config(model_config['first_stem'])
		self.second_stem = set_layer_from_config(model_config['second_stem'])
		self.block_idx += 1
		self.second_stem.drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
		self.stage1 = self._make_stage('stage1')
		self.stage2 = self._make_stage('stage2')
		self.stage3 = self._make_stage('stage3')
		self.stage4 = self._make_stage('stage4')
		self.stage5 = self._make_stage('stage5')
		self.stage6 = self._make_stage('stage6')
		self.feature_mix_layer = set_layer_from_config(model_config['feature_mix_layer'])
		self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

		classifier_config = model_config['classifier']
		classifier_config['out_features'] = num_classes
		self.classifier = set_layer_from_config(classifier_config)

		self._initialization()

	def _get_block_count(self):
		count = 1
		for k in self.model_config.keys():
			if k.startswith('stage'):
				count += len(self.model_config[k])

		return count

	def _make_stage(self, stage_name):
		stage = nn.ModuleList()
		for layer_config in self.model_config[stage_name]:
			self.block_idx += 1
			layer = set_layer_from_config(layer_config)
			layer.drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
			stage.append(layer)

		return stage

	def forward(self, x):
		x = self.first_stem(x)
		x = self.second_stem(x)

		for block in self.stage1:
			x = block(x)
		for block in self.stage2:
			x = block(x)
		for block in self.stage3:
			x = block(x)
		for block in self.stage4:
			x = block(x)
		for block in self.stage5:
			x = block(x)
		for block in self.stage6:
			x = block(x)

		x = self.feature_mix_layer(x)
		x = self.global_avg_pooling(x)
		x = x.view(x.size(0), -1)
		if self.dropout_rate > 0.0:
			x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.classifier(x)
		
		return x

	def get_lookup_latency(self, x):
		if not self.lat_lookup:
			return 0.0

		lat = self.lat_lookup['base']
		x = self.first_stem(x)
		x = self.second_stem(x)

		for block in self.stage1:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage2:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage3:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage4:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage5:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)
		for block in self.stage6:
			key = '{}_{}_{}_{}_{}_k{}_s{}_{}'.format(
												block.name,
												x.size(-1),
												block.in_channels,
												block.se_channels,
												block.out_channels,
												block.kernel_size,
												block.stride,
												block.act_func)
			lat += self.lat_lookup[key][block.mid_channels]
			x = block(x)

		return lat

	@property
	def config(self):
		return {
			'first_stem':  self.first_stem.config,
			'second_stem': self.second_stem.config,
			'stage1': [block.config for block in self.stage1],
			'stage2': [block.config for block in self.stage2],
			'stage3': [block.config for block in self.stage3],
			'stage4': [block.config for block in self.stage4],
			'stage5': [block.config for block in self.stage5],
			'stage6': [block.config for block in self.stage6],
			'feature_mix_layer': self.feature_mix_layer.config,
			'classifier': self.classifier.config,
		}

	def _initialization(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					nn.init.constant_(m.weight, 1)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
