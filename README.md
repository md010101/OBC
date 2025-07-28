## OBC Wisp-implementation README

This repository contains the wisp-compatible implementations for OBC. 

Based on the NeurIPS 2022 paper 
"Optimal Brain Compression: A Framework for Accurate Post-Training Quantization 
and Pruning".

## Files

* `trueobs.py`: efficient implementations of ExactOBS for all compression types
* `main_trueobs.py`: code to run ExactOBS 
* `post_proc.py`: post processing operations like statistics corrections
* `database.py`: generating databases for non-uniform compression
* `spdy.py`: implementation of the DP algorithm for finding non-uniform
  compression configurations; adapted from code provided by the authors of SPDY [9]
* `modelutils.py`: model utilities
* `datautils.py`: data utilities
* `quant.py`: quantization utilities

NOTE: The code as provided here only fully supports torchvision ResNet variants
(the full integration of YOLO and BERT models is omitted due to large amounts
of complex dependencies).

## Usage 

### Dense Checkpoint

First, ensure you have the `.pth` file for the dense checkpoint downloaded.

For example, to download the **ResNet-50 checkpoint pretrained on ImageNet-1K**, run:

```bash
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
```

Then, make sure ImageNet is located/linked to `../imagenet` (alternatively,
you can specifiy the `--datapath` argument for all commands).

### Applying OBC

```
# Quantize weights and activations
python main_trueobs.py rn18 imagenet quant --wbits 4 --abits 4 --save rn18_4w4a.pth

# Prune to the N:M pattern
python main_trueobs.py rn18 imagenet nmprune --prunen 2 --prunem 4 --save rn18_24.pth

# Generate an unstructured pruning database
mkdir models_unstr
python main_trueobs.py rn18 imagenet unstr --sparse-dir models_unstr

# Generate a 4-block pruning database
mkdir models_4block
python main_trueobs.py rn18 imagenet blocked --sparse-dir models_blocked

# Quantize a 2:4 pruned model
python main_trueobs.py rn18 imagenet quant --wbits 4 --abits 4 --load rn18_24.pth --save rn18_24_4w4a.pth 
```

# Statistics Corrections

```
# Batchnorm tuning
python postproc.py rn18 imagenet rn18_24.pth --bnt

# Statistics correction
python postproc.py rn18 imagenet rn18_24.pth --statcorr --statcorr-samples 1024
```

# Non-Uniform Compression

```
mkdir scores

# Unstructured pruning

# Setup database
mkdir models_unstr
python main_trueobs.py rn18 imagenet unstr --sparse-dir models_unstr
# Compute corresponding losses
python database.py rn18 imagenet unstr loss
# Run DP algorithm to determine per-layer compression targets 
python spdy.py rn18 imagenet 2 unstr --dp 
# Stitch profile, apply batchnorm resetting and compute validation accuracy 
python postproc.py rn18 imagenet rn18_unstr_200x_dp.txt --database unstr --bnt

# Mixed quantization + 2:4 pruning

mkdir models_nm
mkdir models_quant
mkdir models_nm_quant
python main_trueobs.py rn18 imagenet nmprune --save models_nm/rn18_24.pth
python main_trueobs.py rn18 imagenet quant --wbits 8 --abits 8 --save models_quant/rn18_8w8a.pth
python main_trueobs.py rn18 imagenet quant --wbits 4 --abits 4 --save models_quant/rn18_4w4a.pth
python main_trueobs.py rn18 imagenet quant --wbits 8 --abits 8 --load models_nm/rn18_24.pth --save models_nm_quant/rn18_24_8w8a.pth 
python main_trueobs.py rn18 imagenet quant --wbits 4 --abits 4 --load models_nm/rn18_24.pth --save models_nm_quant/rn18_24_4w4a.pth 
python database.py rn18 imagenet mixed loss
python spdy.py rn18 imagenet 8 mixed --dp
python postproc.py rn18 imagenet rn18_mixed_800x_dp.txt --database mixed --bnt
```
## Verification

After applying your optimizations, you can verify the sparsity and ImageNet validation accuracy of your pruned ResNet-50 models using the provided verification utility.

### Usage

```bash
python verify_sparse_accuracy.py --ckpt <path_to_checkpoint> --datapath <path_to_imagenet>
```
# Verify a pruned ResNet-50 model
python verify_sparse_accuracy.py \
       --ckpt rn50_24.pth \
       --datapath ../imagenet

# Verify with custom batch size and workers
python verify_sparse_accuracy.py \
       --ckpt rn50_unstr_75sparse.pth \
       --datapath ../imagenet \
       --batch 128 \
       --workers 8
      
Note: This verification utility currently supports ResNet-50 models only.

# How to Download ImageNet-1K from imagenet_utils

This explains how to download & process ImageNet-1K train/val dataset for using as a dataset. Due to legal restrictions, ImageNet-1K cannot be downloaded automatically by script and must be obtained manually. The method below outlines, to our best knowledge, the fastest and most time-efficient way to prepare the dataset. Please note that processing and extraction can take approximately 2–4 hours.


## 1. Data Download

- Download ImageNet-1K train/val dataset from academic torrents : [train link](https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2), [val link](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5)
- Check-out my velog post for download on linux server : [link](https://velog.io/@jasonlee1995/Linux-Server-Download-ImageNet-1K)
- Check-out more informations on original ImageNet website : [link](https://image-net.org/index.php)


## 2. Data Processing
### 2.1. About Data
#### 2.1.1. ImageNet-1K Train Dataset

- ImageNet-1K train dataset zip contains zips like below

```bash
└── ILSVRC2012_img_train.tar
    ├── n01440764.tar
    ├── n01443537.tar
    ├── n01484850.tar
    ├── ...
    └── n15075141.tar
```


### 2.1.2. ImageNet-1K Val Dataset

- ImageNet-1K val dataset zip contains images like below

```bash
└── ILSVRC2012_img_val.tar
    ├── ILSVRV2012_val_00000001.JPEG
    ├── ILSVRV2012_val_00000002.JPEG
    ├── ILSVRV2012_val_00000003.JPEG
    ├── ...
    └── ILSVRV2012_val_00050000.JPEG
```



### 2.2. Files Explain

- `ImageNet_class_index.json` : include class infos
  - **Caution** : same label with different class num exists
    - crane : 134, 517
    - maillot : 638, 639
- `ImageNet_val_label.txt` : include validation image label
- `check.py` : check if unpacked right or not
- `unpack.py` : make clean file trees of `ILSVRC2012_img_train.tar`, `ILSVRC2012_img_val.tar` for using as a dataset


### 2.3. Run
---

1. Assume all the required files are in same directory like below (base_dir)

```bash
└── base_dir
    ├── ILSVRC2012_img_train.tar
    ├── ILSVRC2012_img_val.tar
    ├── ImageNet_class_index.json
    └── ImageNet_val_label.txt
```

---

2. From `unpack.py`, change `base_dir` and `target_dir` variables

---

3. Run `unpack.py` and it makes file trees in specific directory like below (target_dir)

```bash
└── target_dir
    ├── train
    │   ├── 0
    │   │   ├── n01440764_18.JPEG
    │   │   ├── n01440764_36.JPEG
    │   │   └── ...
    │   ├── 1
    │   ├── ...
    │   └── 999
    └── val
        ├── 0
        │   ├── ILSVRC2012_val_00000293.JPEG
        │   ├── ILSVRC2012_val_00002138.JPEG
        │   └── ...
        ├── 1
        ├── ...
        └── 999
```

---

4. From `check.py`, change `ImageNet_dir` variable and run for double-check 

<img width="350" alt="image" src="https://user-images.githubusercontent.com/49643709/163708613-da5fd5e3-2ab2-442a-8028-b9ef20ad7880.png">

---
