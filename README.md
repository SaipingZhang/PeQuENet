# PeQuENet
PeQuENet: Perceptual Quality Enhancement of Compressed Video with Adaptation- and Attention-based Network

## 1. Environment

- Pytorch 1.6.0, CUDA 10.1, Python 3.7

## 2. Dataset

### 2.1. Open-source dataset. 

We use the open-source dataset used in [MFQE 2.0](https://github.com/RyanXingQL/MFQEv2.0). 

Please download raw videos [here](https://github.com/RyanXingQL/MFQEv2.0/wiki/MFQEv2-Dataset). (108 video sequences for training and 18 video sequences for testing) 

Please compress all video sequences by H.265/HEVC reference software HM16.5 under Low Delay P (LDP) configuration with QPs set to be 22, 27, 32 and 37.
  
Please establish two folders named "gt" (ground truth) and "lq" (low quality), respectively. Rename all videos for training, like "QP**_video**_gt" and "QP**_video**_lq". Note that "QP22_video**_gt", "QP27_video**_gt", "QP32_video**_gt" and "QP37_video**_gt" are actually the same. We just repeatly rename the same raw video** four times. While "QP22_video**_lq", "QP27_video**_lq", "QP32_video**_lq" and "QP37_video**_lq" are four low quality versions of the raw video**. Make sure the raw video and the compressed video (in low quality) are one-to-one corresponding. Put all raw videos in the "gt" folder, and put all compressed videos in the "lq" folder. For example,
 
 ```tex
gt/
├── QP22_video_name1_gt
├── QP22_video_name2_gt
...
├── QP22_video_name108_gt
├── QP27_video_name1_gt
...
├── QP37_video_name108_gt

lq/
├── QP22_video_name1_lq
├── QP22_video_name2_lq
...
├── QP22_video_name108_lq
├── QP27_video_name1_lq
...
├── QP37_video_name108_lq
```

Now we have 432 videos in the gt folder and 432 videos in the lq folder. Videos are one-to-one corresponding.

### 2.2. Generate "LMDB" files.

We refer to the way to "generate LMDB" described in [Pytorch implementation of STDF](https://github.com/RyanXingQL/STDF-PyTorch) to generate our own training dataset. Use LMDB files can speed up IO during training.

```
python dataset/create_lmdb_mfqev2.py --opt_path 3frames_mfqev2_1G.yml
```
