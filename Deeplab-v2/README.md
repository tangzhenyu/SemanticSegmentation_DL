## Introduction  

Deeplab v2 + VGGNet16
Deeplab v2 + ResNet 101 or 50

## Dependency
Python 3.5
tensorflow-gpu >= 1.2.0

## Training and Testing

### Prepare Data
# augmented PASCAL VOC
```
mkdir -p ~/DL_dataset
cd ~/DL_dataset       #save datasets $DATASETS
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz # 1.3 GB
tar -zxvf benchmark.tgz
mv benchmark_RELEASE VOC_aug

# original PASCAL VOC 2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # 2 GB
tar -xvf VOCtrainval_11-May-2012.tar
mv VOCdevkit/VOC2012 VOC2012_orig && rm -r VOCdevkit
```

#### Start training

After configuring the network, we can start to train. Run
python main.py
The training of Deeplab v2 ResNet will start.

#### Visualization

We employ tensorboard for visualization.

tensorboard --logdir=logs --port=6006

To visualize the loss curve of training or testing, you can write your own script to make use of the training log.

#### Testing

Change the valid_step in main.py with the checkpoint you want to test. 
You should change valid_num_steps and valid_data_list accordingly. 

python main.py --option=test

The final output includes pixel accuracy and mean IoU.

#### Prediction


python main.py --option=predict
