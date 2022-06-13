# Final Project - Task 1
Pengwei Song & Yifan Qi

## Model & Data
We use a mask R-CNN trained on Cityscapes from mmdetection to do inference on video. The model and config file can be downloaded from [mmdet](https://github.com/open-mmlab/mmdetection/tree/master/configs/cityscapes).

The test video can be downloaded from [video.mp4](https://github.com/open-mmlab/mmdetection/blob/master/demo/demo.mp4).

## Test
```
python test.py --file video.mp4 --out new_video.mp4
```
