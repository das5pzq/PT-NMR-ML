# Scripts

## `area.py` ##

This script is specific for Spin-1/2 sample data. It uses a 2-layer neural network with a flag to switch to training a linear ridge-regression model. Both perform exceptionally well, with the neural network outperforming the RR model. 

## `pol_cnn.py` ##

This script implements the CNN architecture described within the paper for Spin-1 non-cubic symmetric materials. Hyperparameters can vary, but the overall architecture is implemented here. An optional flag to turn on and off the Squeeze-and-Excitation block (described in [Hu's paper](https://arxiv.org/abs/1709.01507)) exists at the beginning for whether or not to use it for low-polarization / high-polarization. Additionally, [residual connections](https://arxiv.org/abs/1512.03385) are implemented alongside an [inception](https://arxiv.org/abs/1409.4842)-style block.

## `pol_mlp.py` ##    

This script implements a simple 2-layer neural network for Spin-1 non-cubic symmetric materials designed primarily for the higher polarization range (2\\% - 60\\%) of ND3, primarily because the relationship between the lineshape and polarization is far more linear than when nearning the TE region ($P_{TE}$ ~ 0.05% for ND3 at B = 5T), though it can also be used for events near TE. 

## `dae.py` ##   

This script implements a [Denoising Autoencoder (DAE)](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) designed to ``denoise" a given Spin-1 signal. The script has currently only been used on Spin-1 simulated lineshape data by itself, not including a varied baseline, though it would be easy to extend the training to simulated data that with those characteristics, as well as Spin-1/2 sample events.
