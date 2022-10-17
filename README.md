# IGA
The source code for the paper "Robust Data Hiding Using Inverse Gradient Attention".

https://arxiv.org/abs/2011.10850

![](/figs/iga_framework.png)

## Installation

Python 3.6.7

torch 1.0.0

torchvision 0.2.1

numpy 1.19.4

## Data Preparation

For coco dataset, we use 10,000 images for training and 1,000 images for validation. Thus, we chose 
those 10,000 + 1,000 images randomly from one of the coco datasets.  http://cocodataset.org/#download.

For DIV2K dataset, we use 800 images for training and 100 images for validation, and it can be downloaded from https://data.vision.ee.ethz.ch/cvl/DIV2K/.


The data directory has the following structure:
```
<data_root>/
  train/
    train_class/
      train_image1.jpg
      train_image2.jpg
      ...
  val/
    val_class/
      val_image1.jpg
      val_image2.jpg
      ...
```

```train_class``` and ```val_class``` folders are so that we can use the standard torchvision data loaders without change.

## Model Running

By default, you can run iga with identity settings. For instance,
```
python -u main.py new -d ../data_path/coco -b 32 -m 90 -r 30 --name iga_identity
```
You can also run iga with combined noises settings. For instance,
```
python -u main.py new -d ../data_path/coco -b 32 -m 90 -r 30 --noise "crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()" --name iga_cn
```


## Citation
If this repository is useful for your research, please consider citing our paper:

```
@article{zhang2020iga,
  title     = {Robust Data Hiding Using Inverse Gradient Attention},
  author    = {Honglei Zhang, Hu Wang, Yuanzhouhan Cao, Chunhua Shen and Yidong Li},
  journal   = {arXiv preprint arXiv:2011.10850},
  year      = {2020}
}
```
