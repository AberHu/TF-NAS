# TF-NAS

Official Pytorch code of paper [TF-NAS: Rethinking Three Search Freedoms of Latency-Constrained Differentiable Neural Architecture Search](https://arxiv.org/abs/2008.05314) in ECCV2020.

With the flourish of differentiable neural architecture search (NAS), automatically searching latency-constrained architectures gives a new perspective to reduce human labor and expertise. However, the searched architectures are usually suboptimal in accuracy and may have large jitters around the target latency. In this paper, we rethink three freedoms of differentiable NAS, i.e. operation-level, depth-level and width-level, and propose a novel method, named Three-Freedom NAS (TF-NAS), to achieve both good classification accuracy and precise latency constraint. For the operation-level, we present a **bi-sampling** search algorithm to moderate the operation collapse. For the depth-level, we introduce a **sink-connecting** search space to ensure the mutual exclusion between skip and other candidate operations, as well as eliminate the architecture redundancy. For the width-level, we propose an **elasticity-scaling** strategy that achieves precise latency constraint in a progressively fine-grained manner. Experiments on ImageNet demonstrate the effectiveness of TF-NAS. Particularly, our searched TF-NAS-A obtains 76.9% top-1 accuracy, achieving state-of-the-art results with less latency. The total search time is only 1.8 days on 1 Titan RTX GPU.

![Overall_Framework](https://github.com/AberHu/TF-NAS/blob/master/images/overall_framework.png)

## Requirements
- Python 3.7
- Pytorch >= 1.1.0
- torchvision >= 0.3.0
- (Optional) apex from [this link](https://github.com/NVIDIA/apex.git)

## Model Zoo
Our pretrained models can be downloaded in the following links. The complete list of the models can be found [here](https://drive.google.com/drive/folders/1IVI33i8KQXPxunNBIObkwK4PjvtwaI-R).

Name | FLOPs | Top-1(%) | Top-5(%) | GPU Lat | CPU Lat | Pretrained Model
:---- | :-----: | :----: | :----: | :----: | :----: | :----:
TF-NAS-A | 457M | 76.87 | 93.11 | 18.03ms | 80.14ms | [Google Drive](https://drive.google.com/drive/folders/1yirUt1X4sxC9-z6g6NyvhHSVvCN7Smx6)
TF-NAS-B | 361M | 76.28 | 92.88 | 15.06ms | 72.10ms | [Google Drive](https://drive.google.com/drive/folders/18PL8Bt8knEuHJgkNuHvLogI8kQckz-jK)
TF-NAS-C | 284M | 75.15 | 92.13 | 11.95ms | 51.87ms | [Google Drive](https://drive.google.com/drive/folders/1S_Hu-bqeMYFaB9YBVBaRcY8z2jQyYbjp)
TF-NAS-D | 219M | 74.19 | 91.45 | 10.08ms | 46.09ms | [Google Drive](https://drive.google.com/drive/folders/18_bXA-79gAVqpPUuu_Qm34JOlnSr6dU4)
TF-NAS-CPU-A | 305M | 75.83 | 92.57 | 14.00ms | 60.11ms | [Google Drive](https://drive.google.com/drive/folders/1GnwPF2W5vY85Lr5nQW2ehF9OaftKEL9E)
TF-NAS-CPU-B | 230M | 74.44 | 91.82 | 10.29ms | 40.09ms | [Google Drive](https://drive.google.com/drive/folders/1NGOXxP9y5z9IYELAnIkP-8hrCn-i_OvF)

GPU and CPU Lat is measured on Titan RTX 24G GPU and Intel Xeon Gold 6130 @ 2.10GHz, respectively.

## Search
For searching, taking the following script as an example:
```
CUDA_VISIBLE_DEVICES=0 python -u train_search.py \
	--img_root "Your ImageNet Train Set Path" \
	--train_list "./dataset/ImageNet-100-effb0_train_cls_ratio0.8.txt" \
	--val_list "./dataset/ImageNet-100-effb0_val_cls_ratio0.8.txt" \
	--lookup_path "./latency_pkl/latency_gpu.pkl" \
	--target_lat 15.0
```
- For GPU latency, set `--lookup_path` to `./latency_pkl/latency_gpu.pkl`. For CPU latency, set `--lookup_path` to `./latency_pkl/latency_cpu.pkl`.
- You can search with different target latencies by changing `--target_lat`.
Please refer to [example.sh](https://github.com/AberHu/TF-NAS/blob/master/example.sh) for more details.

After searching, you can parse the searched architecture by:
```
CUDA_VISIBLE_DEVICES=3 python -u parsing_model.py \
	--model_path "Searched Model Path" \
	--save_path "./model.config" \
	--lookup_path "./latency_pkl/latency_gpu.pkl"
```
You will get a model config file for training and testing, as well as some model profile information. 

## Train
If apex is not installed, please employ [train_eval.py](https://github.com/AberHu/TF-NAS/blob/master/train_eval.py).

- Set `--model_path` to "Searched Model Path". It will parse and train the searched architecture.
```
CUDA_VISIBLE_DEVICES=0,1 python -u train_eval.py \
	--train_root "Your ImageNet Train Set Path" \
	--val_root "Your ImageNet Val Set Path" \
	--train_list "ImageNet Train List" \
	--val_list "ImageNet Val List" \
	--model_path "Searched Model Path"
```
- Or set `--config_path` to the parsed model config file.
```
CUDA_VISIBLE_DEVICES=0,1 python -u train_eval.py \
	--train_root "Your ImageNet Train Set Path" \
	--val_root "Your ImageNet Val Set Path" \
	--train_list "ImageNet Train List" \
	--val_list "ImageNet Val List" \
	--config_path "./model.config"
```

If apex is installed, please employ [train_eval_amp.py](https://github.com/AberHu/TF-NAS/blob/master/train_eval_amp.py). We highly recommend to use mixed precision and distributed training in apex.

- Automatic Mixed Precision
```
CUDA_VISIBLE_DEVICES=0,1 python -u train_eval_amp.py \
	--train_root "Your ImageNet Train Set Path" \
	--val_root "Your ImageNet Val Set Path" \
	--train_list "ImageNet Train List" \
	--val_list "ImageNet Val List" \
	--config_path "./model.config" \
	--opt_level "O1"
```
- Automatic Mixed Precision + DistributedDataParallel
```
CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --nproc_per_node=2 train_eval_amp.py \
	--train_root "Your ImageNet Train Set Path" \
	--val_root "Your ImageNet Val Set Path" \
	--train_list "ImageNet Train List" \
	--val_list "ImageNet Val List" \
	--config_path "./model.config" \
	--opt_level "O1"
```

Please refer to [example.sh](https://github.com/AberHu/TF-NAS/blob/master/example.sh) for more details.

## Test
After training, you can test the trained model by:
```
CUDA_VISIBLE_DEVICES=0 python -u test.py \
	--val_root "Your ImageNet Val Set Path" \
	--val_list "ImageNet Val List" \
	--model_path "./model.config" \
	--weights "Pretrained Weights"
```

## Other
If you are interested in ImageNet training or want to try more tricks, schedulers and properties, please browse [this repo](https://github.com/AberHu/ImageNet-training).

## License
TF-NAS is released under the MIT license. Please see the [LICENSE](https://github.com/AberHu/TF-NAS/blob/master/LICENSE) file for more information.

## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@inproceedings{Hu2020TFNAS,
  title     =  {TF-NAS: Rethinking Three Search Freedoms of Latency-Constrained Differentiable Neural Architecture Search},
  author    =  {Yibo Hu, Xiang Wu and Ran He},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}
```
