# TexRepNet

This code computes a color function that maps positions to colors.
It is used to compute per vertex colors of a 3D reconstruction where color observations are available.
As input you have to provide position-maps of the target scene along with the color observations.
The 'color function' is a per position MLP.
We use the positional encoding proposed by NeRF [(Arxiv)](https://arxiv.org/abs/2003.08934) as well as the basic MLP network structure.

## Training

Given position maps you can start training.
In the figure below you see the training curve for a scan with 89 images (from left to right: position input, predicted texture, ground truth).
![Training](snapshots/train_curve.jpg)

## Inference

Inference is done on a vertex level. You can increase the sample rate via subdivision.
<img src="snapshots/snapshot00_L00.jpg" alt="Matterport1" width="400"/>
<img src="snapshots/snapshot01_L00.jpg" alt="Matterport2" width="400"/>

## Ackowledgements
This code is based on the Pix2Pix/CycleGAN framework [(Github repo)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and the NeuralTexGen project [(Github repo)](https://github.com/JustusThies/NeuralTexGen).
The positional encoding is based on the implementation of nerf-pytorch [(Github repo)](https://github.com/yenchenlin/nerf-pytorch).
Data of Matterport3D [(Github repo)](https://niessner.github.io/Matterport/) is used for demonstration.