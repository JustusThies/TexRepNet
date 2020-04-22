# TexRepNet

This code computes a color function that maps positions to colors.
It is used to compute per vertex colors of a 3D reconstruction where color observations are available.
As input you have to provide position-maps of the target scene along with the color observations.
The 'color function' is a per position MLP.
We use the positional encoding proposed by NeRF [(Arxiv)](https://arxiv.org/abs/2003.08934) as well as the basic MLP network structure.

## Ackowledgements
This code is based on the Pix2Pix/CycleGAN framework [(Github repo)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and the NeuralTexGen project [(Github repo)](https://github.com/JustusThies/NeuralTexGen).
The positional encoding is based on the implementation of nerf-pytorch [(Github repo)](https://github.com/yenchenlin/nerf-pytorch).