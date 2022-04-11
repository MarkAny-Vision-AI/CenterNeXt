# CenterNeXt

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

The official implementation of ["CenterNeXt: Revisiting CenterNet in 2022"]() 

**It has not been published yet.**

## Results and Pre-trained Models

### VOC2007
| backbone | resolution | mAP | FPS(on Titan Xp)| FPS(on RTX 3090) | FLOPs<br>(G) | model config | weights |
|:---|:---:|:---:|:---:| :---:|:---:|:---:|:---:|
| ResNet-18 | 512 x 512 |  74.92 | 110 | 164 | 14.8 | [config](./configs/models/r18_s4.yaml) |[model](https://github.com/MarkAny-Vision-AI/CenterNeXt/releases/download/v0.0.1/r18_s4_best_mAP.pth) |
| + Coupled head | 512 x 512  | 74.61 | 112 | 167 | 14.2 | [config](./configs/models/r18_s4_coupled.yaml) |[model](https://github.com/MarkAny-Vision-AI/CenterNeXt/releases/download/v0.0.1/r18_s4_coupled_best_mAP.pth) |
| + Detection on lower resolution | 512 x 512 | 74.36 | 127 | 191 | 13.0 | [config](/configs/models/r18_s8_coupled.yaml) |[model](https://github.com/MarkAny-Vision-AI/CenterNeXt/releases/download/v0.0.1/r18_s8_coupled_best_mAP.pth) |
| + Mosaic augmentation  | 512 x 512  | 74.20 | 127 | 191 | 13.0 | [config](/configs/models/r18_s8_coupled_mosaic.yaml) |[model](https://github.com/MarkAny-Vision-AI/CenterNeXt/releases/download/v0.0.1/r18_s8_coupled_mosaic_best_mAP.pth) |
| + Mixup augmentation | 512 x 512  | 75.84 | 127 | 191 | 13.0 | [config](/configs/models/r18_s8_coupled_mosaic_mixup.yaml) |[model](https://github.com/MarkAny-Vision-AI/CenterNeXt/releases/download/v0.0.1/r18_s8_coupled_mosaic_mixup_best_mAP.pth) |
| ResNet-50 | 512 x 512  | 80.46  | 65 | 104 | 25.0 | [config](/configs/models/r50.yaml) |[model](https://github.com/MarkAny-Vision-AI/CenterNeXt/releases/download/v0.0.1/r50_best_mAP.pth) |
| ResNet-101 | 512 x 512  | 83.00  | 39 | 60 | 44.5 | [config](/configs/models/r101.yaml) |[model](https://github.com/MarkAny-Vision-AI/CenterNeXt/releases/download/v0.0.1/r101_best_mAP.pth) |
| ConvNeXt-T | 512 x 512  | 83.57  | 43 | 87 | 26.8 | [config](/configs/models/convnext-t.yaml) |[model](https://github.com/MarkAny-Vision-AI/CenterNeXt/releases/download/v0.0.1/convnext-t_best_mAP.pth) |

## Setup
Create a new conda virtual environment

```
conda create -n centernext python=3.8 -y
conda activate centernext
```

Install Pytorch and torchvision following official instructions. For example:

```
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Clone this repo and install required packages:
```
git clone https://github.com/MarkAny-Vision-AI/CenterNeXt
cd CenterNeXt
pip install -r requirements.txt
```

Download a dataset:

if (your_os == 'Window'):
```
cd CenterNeXt
scripts/download-voc0712.bat
python create_gt_files.py
```
else:
```
cd CenterNeXt
scripts/download-voc0712.sh
python create_gt_files.py
```

### Training
```
python train.py --model ./configs/models/your_model.yaml 
```

#### On Multi GPU
```
torchrun --nproc_per_node={number of gpus} train.py --model ./configs/models/your_model.yaml 
```


### Evaluation
```
python eval.py --model ./configs/models/your_model.yaml --weights /path/to/your_model.pth
```

### Profiling
```
python profile.py --model ./configs/models/your_model.yaml
```
## License

This project is licensed under the terms of the **Attribution-NonCommercial 4.0 International license**.
It is released for academic research only and is free to researchers from educational or research institutes for **non-commercial purposes**. 

Please see the [LICENSE](./LICENSE) file for more information.

Please contact contentsrnd@markany.com or works@markany.com for business inquiries.
