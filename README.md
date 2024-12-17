# SLTNet: Efficient Event-based Semantic Segmentation with Spike-driven Lightweight Transformer-based Networks

This repository is an official PyTorch implementation of our paper, which was submitted to **IEEE ICRA 2025**, titled "Efficient Event-based Semantic Segmentation with Spike-driven Lightweight Transformer-based Networks".


## Installation

```
cuda == 10.2
Python == 3.6.4
Pytorch == 1.8.0+cu101

# clone this repository
git clone https://github.com/longxianlei/SLTNet.git

```

## Train

```
# DDD17
python train.py --model SLTNet --dataset DDD17_events --classes 6 --dataset_path your_datasets_path --split train --batch_size 32

# DSEC
python train.py --model SLTNet --dataset DSEC_events --classes 11 --dataset_path' your_datasets_path --split train --batch_size 32
```



## Test

```
# DDD17
python test.py --dataset DDD17_events --checkpoint ./checkpoint/DDD17_events/SLTNet_ddd17_model_best_miou.pth

# DSEC
python test.py --dataset DSEC_events --checkpoint ./checkpoint/DSEC_events/SLTNet_dsec_model_best_miou.pth
```


## Results

- Please refer to our article for more details.

| Methods | Dataset | Input Size | mIoU(%) |
| :-----: | :-----: | :--------: | :-----: |
| LETNet  |  DDD17  |  200x346   |  51.92  |
| LETNet  |  DSEC   |  480x640   |  47.91  |



## Citation

If you find this project useful for your research, please cite our paper:

```
xxx
```

## Thanks && Refer

```bash
@misc{Efficient-Segmentation-Networks,
  author = {Yu Wang},
  title = {Efficient-Segmentation-Networks Pytorch Implementation},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks}},
  commit = {master}
}
```

For more code about lightweight real-time semantic segmentation, please refer to: https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks
