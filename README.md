# PeQuENet
PeQuENet: Perceptual Quality Enhancement of Compressed Video with Adaptation- and Attention-based Network

## Environment

- Pytorch 1.6.0, CUDA 10.1, Python 3.7

## Dataset

- We use the open-source dataset (108 video sequences for training and 18 video sequences for testing) used in [MFQE 2.0](https://github.com/RyanXingQL/MFQEv2.0). 

  Please download raw videos [here](https://github.com/RyanXingQL/MFQEv2.0/wiki/MFQEv2-Dataset).

  Please compress all video sequences by H.265/HEVC reference software HM16.5 under Low Delay P (LDP) configuration with QPs set to be 22, 27, 32 and 37.
  
  please rename all videos for training like "QP**_video**_gt" and "QP**_video**_lq". Note that we just "QP22_video**_gt", "QP27_video**_gt", "QP32_video**_gt" and "QP37_video**_gt" are actually the same.
  
  Please establish two folders named "gt" (ground truth) and "lq" (low quality), respectively. Put all raw videos in the "gt" folder, and put all compressed videos in the "lq"  folder.
  
  
  MFQEv2_dataset/
├── train_108/
│   ├── raw/
│   └── HM16.5_LDP/
│       └── QP37/
├── test_18/
│   ├── raw/
│   └── HM16.5_LDP/
│       └── QP37/
├── mfqev2_train_gt.lmdb/
└── mfqev2_train_lq.lmdb/


- We refer to the way to "edit YML" and "generate LMDB" described in [Pytorch implementation of STDF](https://github.com/RyanXingQL/STDF-PyTorch) to generate our own training dataset.
