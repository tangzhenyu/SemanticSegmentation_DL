## Introduction
RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation

A tensorflow implement of refinenet discribed in [arxiv:1611.06612](https://arxiv.org/abs/1611.06612).

## prepare
- download the pretrain model of resnet_v1_101.ckpt, you can download it from [here](https://github.com/tensorflow/models/tree/master/slim)
- download the [pascal voc 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

## training
- Run convert_pascal_voc_to_tfrecords.py to convert training data into .tfrecords file.
- Run python nets/multi_gpu_train.py.

## eval
- Pretrained on pascal voc.[model](https://pan.baidu.com/s/1qXW9OPA),Pass:772v.
- put images in demo/ and run python nets/demo.py 

