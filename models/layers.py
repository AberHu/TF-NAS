import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

sys.path.append('..')
from tools.utils import *

def set_layer_from_config(layer_config):
	if layer_config is None:
		return None

	name2layer = {
		ConvLayer.__name__: ConvLayer,
		IdentityLayer.__name__: IdentityLayer,
		LinearLayer.__name__: LinearLayer,
		MBInvertedResBlock.__name__: MBInvertedResBlock,
	}

	layer_name = layer_config.pop('name')
	layer = name2layer[layer_name]
	return layer.build_from_config(layer_config)


class Swish(nn.Module):
	def __init__(self, inplace=False):
		super(Swish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		if self.inplace:
			return x.mul_(x.sigmoid())
		else:
			return x * x.sigmoid()


class HardSwish(nn.Module):
	def __init__(self, inplace=False):
		super(HardSwish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		if self.inplace:
			return x.mul_(F.relu6(x + 3., inplace=True) / 6.)
		else:
			return x * F.relu6(x + 3.) /6.


class BasicUnit(nn.Module):

	def forward(self, x):
		raise NotImplementedError

	@property
	def name(self):
		raise NotImplementedError

	@property
	def unit_str(self):
		raise NotImplementedError

	@property
	def config(self):
		raise NotImplementedError

	@staticmethod
	def build_from_config(config):
		raise NotImplementedError
	
	def get_flops(self, x):
		raise NotImplementedError

	def get_latency(self, x):
		raise NotImplementedError


class BasicLayer(BasicUnit):

	def __init__(
			self,
			in_channels,
			out_channels,
			use_bn=True,
			affine = True,
			act_func='relu6',
			ops_order='weight_bn_act'):
		super(BasicLayer, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.use_bn = use_bn
		self.affine = affine
		self.act_func = act_func
		self.ops_order = ops_order

		""" add modules """
		# batch norm
		if self.use_bn:
			if self.bn_before_weight:
				self.bn = nn.BatchNorm2d(in_channels, affine=affine, track_running_stats=affine)
			else:
				self.bn = nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=affine)
		else:
			self.bn = None
		# activation
		if act_func == 'relu':
			if self.ops_list[0] == 'act':
				self.act = nn.ReLU(inplace=False)
			else:
				self.act = nn.ReLU(inplace=True)
		elif act_func == 'relu6':
			if self.ops_list[0] == 'act':
				self.act = nn.ReLU6(inplace=False)
			else:
				self.act = nn.ReLU6(inplace=True)
		elif act_func == 'swish':
			if self.ops_list[0] == 'act':
				self.act = Swish(inplace=False)
			else:
				self.act = Swish(inplace=True)
		elif act_func == 'h-swish':
			if self.ops_list[0] == 'act':
				self.act = HardSwish(inplace=False)
			else:
				self.act = HardSwish(inplace=True)
		else:
			self.act = None

	@property
	def ops_list(self):
		return self.ops_order.split('_')

	@property
	def bn_before_weight(self):
		for op in self.ops_list:
			if op == 'bn':
				return True
			elif op == 'weight':
				return False
		raise ValueError('Invalid ops_order: %s' % self.ops_order)

	def weight_call(self, x):
		raise NotImplementedError

	def forward(self, x):
		for op in self.ops_list:
			if op == 'weight':
				x = self.weight_call(x)
			elif op == 'bn':
				if self.bn is not None:
					x = self.bn(x)
			elif op == 'act':
				if self.act is not None:
					x = self.act(x)
			else:
				raise ValueError('Unrecognized op: %s' % op)
		return x

	@property
	def name(self):
		raise NotImplementedError

	@property
	def unit_str(self):
		raise NotImplementedError

	@property
	def config(self):
		return {
			'in_channels': self.in_channels,
			'out_channels': self.out_channels,
			'use_bn': self.use_bn,
			'affine': self.affine,
			'act_func': self.act_func,
			'ops_order': self.ops_order,
		}

	@staticmethod
	def build_from_config(config):
		raise NotImplementedError

	def get_flops(self):
		raise NotImplementedError

	def get_latency(self):
		raise NotImplementedError


class ConvLayer(BasicLayer):

	def __init__(
			self,
			in_channels,
			out_channels,
			kernel_size=3,
			stride=1,
			groups=1,
			has_shuffle=False,
			bias=False,
			use_bn=True,
			affine=True,
			act_func='relu6',
			ops_order='weight_bn_act'):
		super(ConvLayer, self).__init__(
			in_channels,
			out_channels,
			use_bn,
			affine,
			act_func,
			ops_order)

		self.kernel_size = kernel_size
		self.stride = stride
		self.groups = groups
		self.has_shuffle = has_shuffle
		self.bias = bias

		padding = get_same_padding(self.kernel_size)
		self.conv = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size=self.kernel_size,
			stride=self.stride,
			padding=padding,
			groups=self.groups,
			bias=self.bias)

	def weight_call(self, x):
		x = self.conv(x)
		if self.has_shuffle and self.groups > 1:
			x = channel_shuffle(x, self.groups)
		return x

	@property
	def name(self):
		return ConvLayer.__name__

	@property
	def unit_str(self):
		if isinstance(self.kernel_size, int):
			kernel_size = (self.kernel_size, self.kernel_size)
		else:
			kernel_size = self.kernel_size
		if self.groups == 1:
			return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
		else:
			return '%dx%d_GroupConv_G%d' % (kernel_size[0], kernel_size[1], self.groups)

	@property
	def config(self):
		config = {
			'name': ConvLayer.__name__,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'groups': self.groups,
			'has_shuffle': self.has_shuffle,
			'bias': self.bias,
		}
		config.update(super(ConvLayer, self).config)
		return config

	@staticmethod
	def build_from_config(config):
		return ConvLayer(**config)

	def get_flops(self):
		raise NotImplementedError

	def get_latency(self):
		raise NotImplementedError


class IdentityLayer(BasicLayer):

	def __init__(
			self,
			in_channels,
			out_channels,
			use_bn=False,
			affine=False,
			act_func=None,
			ops_order='weight_bn_act'):
		super(IdentityLayer, self).__init__(
			in_channels,
			out_channels,
			use_bn,
			affine,
			act_func,
			ops_order)

	def weight_call(self, x):
		return x

	@property
	def name(self):
		return IdentityLayer.__name__

	@property
	def unit_str(self):
		return 'Identity'

	@property
	def config(self):
		config = {
			'name': IdentityLayer.__name__,
		}
		config.update(super(IdentityLayer, self).config)
		return config

	@staticmethod
	def build_from_config(config):
		return IdentityLayer(**config)

	def get_flops(self):
		raise NotImplementedError

	def get_latency(self):
		raise NotImplementedError


class LinearLayer(BasicUnit):

	def __init__(
			self,
			in_features,
			out_features,
			bias=True,
			use_bn=False,
			affine=False,
			act_func=None,
			ops_order='weight_bn_act'):
		super(LinearLayer, self).__init__()

		self.in_features = in_features
		self.out_features = out_features
		self.bias = bias
		self.use_bn = use_bn
		self.affine = affine
		self.act_func = act_func
		self.ops_order = ops_order

		""" add modules """
		# batch norm
		if self.use_bn:
			if self.bn_before_weight:
				self.bn = nn.BatchNorm1d(in_features, affine=affine, track_running_stats=affine)
			else:
				self.bn = nn.BatchNorm1d(out_features, affine=affine, track_running_stats=affine)
		else:
			self.bn = None
		# activation
		if act_func == 'relu':
			if self.ops_list[0] == 'act':
				self.act = nn.ReLU(inplace=False)
			else:
				self.act = nn.ReLU(inplace=True)
		elif act_func == 'relu6':
			if self.ops_list[0] == 'act':
				self.act = nn.ReLU6(inplace=False)
			else:
				self.act = nn.ReLU6(inplace=True)
		elif act_func == 'tanh':
			self.act = nn.Tanh()
		elif act_func == 'sigmoid':
			self.act = nn.Sigmoid()
		else:
			self.act = None
		# linear
		self.linear = nn.Linear(self.in_features, self.out_features, self.bias)

	@property
	def ops_list(self):
		return self.ops_order.split('_')

	@property
	def bn_before_weight(self):
		for op in self.ops_list:
			if op == 'bn':
				return True
			elif op == 'weight':
				return False
		raise ValueError('Invalid ops_order: %s' % self.ops_order)

	def forward(self, x):
		for op in self.ops_list:
			if op == 'weight':
				x = self.linear(x)
			elif op == 'bn':
				if self.bn is not None:
					x = self.bn(x)
			elif op == 'act':
				if self.act is not None:
					x = self.act(x)
			else:
				raise ValueError('Unrecognized op: %s' % op)
		return x

	@property
	def name(self):
		return LinearLayer.__name__
	
	@property
	def unit_str(self):
		return '%dx%d_Linear' % (self.in_features, self.out_features)

	@property
	def config(self):
		return {
			'name': LinearLayer.__name__,
			'in_features': self.in_features,
			'out_features': self.out_features,
			'bias': self.bias,
			'use_bn': self.use_bn,
			'affine': self.affine,
			'act_func': self.act_func,
			'ops_order': self.ops_order,
		}

	@staticmethod
	def build_from_config(config):
		return LinearLayer(**config)

	def get_flops(self):
		raise NotImplementedError

	def get_latency(self):
		raise NotImplementedError


class MBInvertedResBlock(BasicUnit):

	def __init__(
			self,
			in_channels,
			mid_channels,
			se_channels,
			out_channels,
			kernel_size=3,
			stride=1,
			groups=1,
			has_shuffle=False,
			bias=False,
			use_bn=True,
			affine=True,
			act_func='relu6'):
		super(MBInvertedResBlock, self).__init__()

		self.in_channels = in_channels
		self.mid_channels = mid_channels
		self.se_channels = se_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.groups = groups
		self.has_shuffle = has_shuffle
		self.bias = bias
		self.use_bn = use_bn
		self.affine = affine
		self.act_func = act_func
		self.drop_connect_rate = 0.0

		# inverted bottleneck
		if mid_channels > in_channels:
			inverted_bottleneck = OrderedDict([
					('conv', nn.Conv2d(in_channels, mid_channels, 1, 1, 0, groups=groups, bias=bias)),
				])
			if use_bn:
				inverted_bottleneck['bn'] = nn.BatchNorm2d(mid_channels, affine=affine, track_running_stats=affine)
			if act_func == 'relu':
				inverted_bottleneck['act'] = nn.ReLU(inplace=True)
			elif act_func == 'relu6':
				inverted_bottleneck['act'] = nn.ReLU6(inplace=True)
			elif act_func == 'swish':
				inverted_bottleneck['act'] = Swish(inplace=True)
			elif act_func == 'h-swish':
				inverted_bottleneck['act'] = HardSwish(inplace=True)
			self.inverted_bottleneck = nn.Sequential(inverted_bottleneck)
		else:
			self.inverted_bottleneck = None
			self.mid_channels = in_channels
			mid_channels = in_channels

		# depthwise convolution
		padding = get_same_padding(self.kernel_size)
		depth_conv = OrderedDict([
				('conv', 
				 nn.Conv2d(
				 	 mid_channels,
				 	 mid_channels,
				 	 kernel_size,
				 	 stride,
				 	 padding,
				 	 groups=mid_channels,
				 	 bias=bias)),
			])
		if use_bn:
			depth_conv['bn'] = nn.BatchNorm2d(mid_channels, affine=affine, track_running_stats=affine)
		if act_func == 'relu':
			depth_conv['act'] = nn.ReLU(inplace=True)
		elif act_func == 'relu6':
			depth_conv['act'] = nn.ReLU6(inplace=True)
		elif act_func == 'swish':
			depth_conv['act'] = Swish(inplace=True)
		elif act_func == 'h-swish':
			depth_conv['act'] = HardSwish(inplace=True)
		self.depth_conv = nn.Sequential(depth_conv)

		# se model
		if se_channels > 0:
			squeeze_excite = OrderedDict([
					('conv_reduce', nn.Conv2d(mid_channels, se_channels, 1, 1, 0, groups=groups, bias=True)),
				])
			if act_func == 'relu':
				squeeze_excite['act'] = nn.ReLU(inplace=True)
			elif act_func == 'relu6':
				squeeze_excite['act'] = nn.ReLU6(inplace=True)
			elif act_func == 'swish':
				squeeze_excite['act'] = Swish(inplace=True)
			elif act_func == 'h-swish':
				squeeze_excite['act'] = HardSwish(inplace=True)
			squeeze_excite['conv_expand'] = nn.Conv2d(se_channels, mid_channels, 1, 1, 0, groups=groups, bias=True)
			self.squeeze_excite = nn.Sequential(squeeze_excite)
		else:
			self.squeeze_excite = None
			self.se_channels = 0

		# pointwise linear
		point_linear = OrderedDict([
				('conv', nn.Conv2d(mid_channels, out_channels, 1, 1, 0, groups=groups, bias=bias)),
			])
		if use_bn:
			point_linear['bn'] = nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=affine)
		self.point_linear = nn.Sequential(point_linear)

		# residual flag
		self.has_residual = (in_channels == out_channels) and (stride == 1)

	def forward(self, x):
		res = x

		if self.inverted_bottleneck is not None:
			x = self.inverted_bottleneck(x)
			if self.has_shuffle and self.groups > 1:
				x = channel_shuffle(x, self.groups)

		x = self.depth_conv(x)
		if self.squeeze_excite is not None:
			x_se = F.adaptive_avg_pool2d(x, 1)
			x = x * torch.sigmoid(self.squeeze_excite(x_se))

		x = self.point_linear(x)
		if self.has_shuffle and self.groups > 1:
			x = channel_shuffle(x, self.groups)

		if self.has_residual:
			if self.drop_connect_rate > 0.0:
				x = drop_connect(x, self.training, self.drop_connect_rate)
			x += res

		return x

	@property
	def name(self):
		return MBInvertedResBlock.__name__
	
	@property
	def unit_str(self):
		if isinstance(self.kernel_size, int):
			kernel_size = (self.kernel_size, self.kernel_size)
		else:
			kernel_size = self.kernel_size
		if self.groups == 1:
			return '%dx%d_MBInvResBlock_E%.2f' % (kernel_size[0], kernel_size[1], 
													self.mid_channels * 1.0 / self.in_channels)
		else:
			return '%dx%d_GroupMBInvResBlock_E%.2f_G%d' % (kernel_size[0], kernel_size[1], 
															self.mid_channels * 1.0 / self.in_channels, self.groups)

	@property
	def config(self):
		return {
			'name': MBInvertedResBlock.__name__,
			'in_channels': self.in_channels,
			'mid_channels': self.mid_channels,
			'se_channels': self.se_channels,
			'out_channels': self.out_channels,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'groups': self.groups,
			'has_shuffle': self.has_shuffle,
			'bias': self.bias,
			'use_bn': self.use_bn,
			'affine': self.affine,
			'act_func': self.act_func,
		}

	@staticmethod
	def build_from_config(config):
		return MBInvertedResBlock(**config)


	def get_flops(self):
		raise NotImplementedError

	def get_latency(self):
		raise NotImplementedError
