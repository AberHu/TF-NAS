from .config import mc_mask_dddict, lat_lookup_key_dddict
from .flops_benchmark import calculate_FLOPs_in_M
from .utils import measure_latency_in_ms, count_parameters_in_MB
from .utils import AverageMeter, accuracy
from .utils import drop_connect, channel_shuffle, get_same_padding
from .utils import create_exp_dir, save_checkpoint