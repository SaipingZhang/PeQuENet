# PeQuENet
PeQuENet: Perceptual Quality Enhancement of Compressed Video with Adaptation- and Attention-based Network

## Environment

- Pytorch 1.6.0, CUDA 10.1, Python 3.7

## Dataset

- We use the open-source dataset (108 video sequences for training and 18 video sequences for testing) used in [MFQE 2.0](https://github.com/RyanXingQL/MFQEv2.0). 

  Please download videos [here](https://github.com/RyanXingQL/MFQEv2.0/wiki/MFQEv2-Dataset).

  Please compress all video sequences by H.265/HEVC reference software HM16.5 under Low Delay P (LDP) configuration with QPs set to be 22, 27, 32 and 37.

- We refer to the way to "edit YML" and "generate LMDB" described in [Pytorch implementation of STDF](https://github.com/RyanXingQL/STDF-PyTorch) to generate our own training dataset.
