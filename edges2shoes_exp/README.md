## Edges-to-Shoes experiment

First you need to prepare data as explained in `datasets` folder

### Training Augmented CycleGAN model
`CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/edges2shoes/ --name augcgan_model`